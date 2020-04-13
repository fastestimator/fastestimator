# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
import tempfile
import numpy as np
from typing import Callable, Iterable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as fn
from transformers import BertTokenizer, BertModel

import fastestimator as fe
from fastestimator.dataset.data import german_ner
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import PadSequence, Tokenize, WordtoId
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from fastestimator.trace.io import BestModelSaver

def char2idx(data):
    tag2idx = {t: i for i, t in enumerate(data)}
    return tag2idx


class AttentionMask(NumpyOp):
    def forward(self, data, state):
        masks = [float(i > 0) for i in data]
        return np.array(masks)


class ReshapeOp(TensorOp):
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = "!infer"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        inp_shape = data.size()
        if len(inp_shape) < 3:
            data = data.view(-1,)
        else:
            data = data.view(-1, inp_shape[2])
        return data
bert_model = BertModel.from_pretrained("bert-base-uncased")

class NERModel(nn.Module):
    def __init__(self, head_masks):
        super().__init__()
        self.head_masks = head_masks
        self.bert_embed = list(bert_model.children())[0]
        self.bert_encode = list(bert_model.children())[1]
        self.fc = nn.Linear(in_features=768, out_features=24)

    def forward(self, inp):
        x, x_masks = inp
        x_masks = x_masks[:, None, None, :]
        x_masks = x_masks.to(dtype=torch.float)
        seq_output = self.bert_embed(x)
        out = self.bert_encode(seq_output, attention_mask=x_masks, head_mask=self.head_masks)
        out = self.fc(out[0])
        out = fn.softmax(out, dim=-1)
        return out


def get_estimator(max_len=20, epochs=10, batch_size=64, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):

    # step 1 prepare data
    train_data, eval_data, data_vocab, label_vocab = german_ner.load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tag2idx = char2idx(label_vocab)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Tokenize(inputs="x", outputs="x", tokenize_fn=tokenizer.tokenize),
            WordtoId(inputs="x", outputs="x", mapping=tokenizer.convert_tokens_to_ids),
            WordtoId(inputs="y", outputs="y", mapping=tag2idx),
            PadSequence(max_len=max_len, inputs="x", outputs="x"),
            PadSequence(max_len=max_len, value=23, inputs="y", outputs="y"),
            AttentionMask(inputs="x", outputs="x_masks")
        ])

    # step 2. prepare model
    head_masks = [None] * 12
    model = fe.build(model_fn=lambda: NERModel(head_masks=head_masks), optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-5))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs=["x", "x_masks"], outputs="y_pred"),
        ReshapeOp(inputs="y", outputs="y"),
        ReshapeOp(inputs="y_pred", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])

    traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=save_dir)]

    # step 3 prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
