# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from transformers import BertTokenizer

import fastestimator as fe
from fastestimator.dataset.data import tednmt
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import LossOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.trace import Trace
from fastestimator.util import get_num_devices


class Encode(NumpyOp):
    def __init__(self, tokenizer, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = tokenizer

    def forward(self, data, state):
        return np.array(self.tokenizer.encode(data))


class ShiftData(TensorOp):
    def forward(self, data, state):
        target = data
        return target[:, :-1], target[:, 1:]


def lr_fn(step, em_dim, warmupstep=4000):
    lr = em_dim**-0.5 * min(step**-0.5, step * warmupstep**-1.5)
    return lr


class MaskedAccuracy(Trace):
    def on_epoch_begin(self, data):
        self.correct = 0
        self.total = 0

    def on_batch_end(self, data):
        y_pred, y_true = data["pred"].numpy(), data["target_real"].numpy()
        mask = np.logical_not(y_true == 0)
        matches = np.logical_and(y_true == np.argmax(y_pred, axis=2), mask)
        self.correct += np.sum(matches)
        self.total += np.sum(mask)

    def on_epoch_end(self, data):
        data.write_with_log(self.outputs[0], self.correct / self.total)


class Transformer(nn.Module):
    def __init__(self, num_layers, em_dim, num_heads, ff_dim, input_vocab, target_vocab, max_pos_enc, max_pos_dec):
        super().__init__()
        self.em_dim = em_dim
        # encoder layers
        self.encode_embedding = nn.Embedding(input_vocab, em_dim)
        self.encode_pos_embedding = PositionalEncoding(max_pos=max_pos_enc, em_dim=em_dim)
        encoder_layer = TransformerEncoderLayer(em_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.encode_dropout = nn.Dropout(p=0.1)
        # decoder layers
        self.decode_embedding = nn.Embedding(target_vocab, em_dim)
        self.decode_pos_embedding = PositionalEncoding(max_pos=max_pos_dec, em_dim=em_dim)
        decoder_layer = TransformerDecoderLayer(em_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.decode_dropout = nn.Dropout(p=0.1)
        self.final_linear = nn.Linear(em_dim, target_vocab)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask):
        # Switch batch and sequence length dimension for pytorch convention
        src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        src_em = self.encode_embedding(src) * math.sqrt(self.em_dim)
        src_em = self.encode_pos_embedding(src_em)
        src_em = self.encode_dropout(src_em)
        encoder_output = self.transformer_encoder(src_em, src_key_padding_mask=src_pad_mask)
        tgt_em = self.decode_embedding(tgt) * math.sqrt(self.em_dim)
        tgt_em = self.decode_pos_embedding(tgt_em)
        tgt_em = self.decode_dropout(tgt_em)
        decoder_output = self.transformer_decoder(tgt_em,
                                                  encoder_output,
                                                  tgt_key_padding_mask=tgt_pad_mask,
                                                  tgt_mask=tgt_mask[0],
                                                  memory_key_padding_mask=src_pad_mask)
        output = self.final_linear(decoder_output)
        output = output.transpose(0, 1)  # Switch back
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, max_pos, em_dim):
        super().__init__()
        self.max_pos = max_pos
        self.em_dim = em_dim
        pe = torch.zeros(max_pos, em_dim)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, em_dim, 2).float() * (-math.log(10000.0) / em_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CreateMasks(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_device = get_num_devices()

    def forward(self, data, state):
        inp, tar = data
        encode_pad_mask = self.create_padding_mask(inp)
        decode_pad_mask = self.create_padding_mask(tar)
        dec_look_ahead_mask = self.create_look_ahead_mask(tar)
        dec_look_ahead_mask = torch.stack([dec_look_ahead_mask for _ in range(self.num_device)])
        return encode_pad_mask, decode_pad_mask, dec_look_ahead_mask

    @staticmethod
    def create_padding_mask(seq):
        return seq == 0

    @staticmethod
    def create_look_ahead_mask(seq):
        return torch.triu(torch.ones(seq.size(1), seq.size(1), dtype=torch.bool), diagonal=1).to(seq.device)


class MaskedCrossEntropy(LossOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, data, state):
        y_pred, y_true = data
        mask = y_true != 0
        loss = self.loss_fn(input=y_pred.reshape(-1, y_pred.size(-1)), target=y_true.reshape(-1)) * mask.reshape(-1)
        loss = torch.sum(loss) / torch.sum(mask)
        return loss


def lr_fn(step, em_dim, warmupstep=4000):
    lr = em_dim**-0.5 * min(step**-0.5, step * warmupstep**-1.5)
    return lr


def get_estimator(data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  epochs=20,
                  em_dim=128,
                  batch_size=32,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None):
    train_ds, eval_ds, test_ds = tednmt.load_data(data_dir, translate_option="pt_to_en")
    pt_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        test_data=test_ds,
        batch_size=batch_size,
        ops=[
            Encode(inputs="source", outputs="source", tokenizer=pt_tokenizer),
            Encode(inputs="target", outputs="target", tokenizer=en_tokenizer)
        ],
        pad_value=0)
    model = fe.build(
        model_fn=lambda: Transformer(num_layers=4,
                                     em_dim=em_dim,
                                     num_heads=8,
                                     ff_dim=512,
                                     input_vocab=pt_tokenizer.vocab_size,
                                     target_vocab=en_tokenizer.vocab_size,
                                     max_pos_enc=1000,
                                     max_pos_dec=1000),
        optimizer_fn="adam")
    network = fe.Network(ops=[
        ShiftData(inputs="target", outputs=("target_inp", "target_real")),
        CreateMasks(inputs=("source", "target_inp"),
                    outputs=("encode_pad_mask", "decode_pad_mask", "dec_look_ahead_mask")),
        ModelOp(model=model,
                inputs=("source", "target_inp", "encode_pad_mask", "decode_pad_mask", "dec_look_ahead_mask"),
                outputs="pred"),
        MaskedCrossEntropy(inputs=("pred", "target_real"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        MaskedAccuracy(inputs=("pred", "target_real"), outputs="masked_acc", mode="!train"),
        BestModelSaver(model=model, save_dir=model_dir, metric="masked_acc", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: lr_fn(step, em_dim))
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             traces=traces,
                             epochs=epochs,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator
