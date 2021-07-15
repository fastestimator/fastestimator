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
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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


class Encode(NumpyOp):
    def __init__(self, tokenizer, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = tokenizer

    def forward(self, data, state):
        return np.array(self.tokenizer.encode(data))


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    scaled_attention_logits += (mask * -1e9)  # this is to make the softmax of masked cells to be 0
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output


def point_wise_feed_forward_network(em_dim, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(em_dim)  # (batch_size, seq_len, em_dim)
    ])


class MultiHeadAttention(layers.Layer):
    def __init__(self, em_dim, num_heads):
        super().__init__()
        assert em_dim % num_heads == 0, "model dimension must be multiply of number of heads"
        self.num_heads = num_heads
        self.em_dim = em_dim
        self.depth = em_dim // self.num_heads
        self.wq = layers.Dense(em_dim)
        self.wk = layers.Dense(em_dim)
        self.wv = layers.Dense(em_dim)
        self.dense = layers.Dense(em_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, num_heads, seq_len, depth

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # B, seq_len, em_dim
        k = self.wk(k)  # B, seq_len, em_dim
        v = self.wv(v)  # B, seq_len, em_dim
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  #B, seq_len, num_heads, depth
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.em_dim))  # B, seq_len, em_dim
        output = self.dense(concat_attention)
        return output


class EncoderLayer(layers.Layer):
    def __init__(self, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(em_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(em_dim, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


def get_angles(pos, i, em_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(em_dim))
    return pos * angle_rates


def positional_encoding(position, em_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(em_dim)[np.newaxis, :], em_dim)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class DecoderLayer(layers.Layer):
    def __init__(self, em_dim, num_heads, diff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(em_dim, num_heads)
        self.mha2 = MultiHeadAttention(em_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(em_dim, diff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_out, training, decode_mask, padding_mask):
        attn1 = self.mha1(x, x, x, decode_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(enc_out, enc_out, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Encoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, input_vocab, max_pos_enc, rate=0.1):
        super().__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.embedding = layers.Embedding(input_vocab, em_dim)
        self.pos_encoding = positional_encoding(max_pos_enc, self.em_dim)
        self.enc_layers = [EncoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.em_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, target_vocab, max_pos_enc, rate=0.1):
        super().__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.embedding = layers.Embedding(target_vocab, em_dim)
        self.pos_encoding = positional_encoding(max_pos_enc, em_dim)
        self.dec_layers = [DecoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, decode_mask, padding_mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.em_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, decode_mask, padding_mask)
        return x


def transformer(num_layers, em_dim, num_heads, dff, input_vocab, target_vocab, max_pos_enc, max_pos_dec, rate=0.1):
    inputs = layers.Input(shape=(None, ))
    targets = layers.Input(shape=(None, ))
    encode_mask = layers.Input(shape=(None, None, None))
    decode_mask = layers.Input(shape=(None, None, None))
    x = Encoder(num_layers, em_dim, num_heads, dff, input_vocab, max_pos_enc, rate=rate)(inputs, encode_mask)
    x = Decoder(num_layers, em_dim, num_heads, dff, target_vocab, max_pos_dec, rate=rate)(targets,
                                                                                          x,
                                                                                          decode_mask,
                                                                                          encode_mask)
    x = layers.Dense(target_vocab)(x)
    model = tf.keras.Model(inputs=[inputs, targets, encode_mask, decode_mask], outputs=x)
    return model


class CreateMasks(TensorOp):
    def forward(self, data, state):
        inp, tar = data
        encode_mask = self.create_padding_mask(inp)
        dec_look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        decode_mask = tf.maximum(dec_target_padding_mask, dec_look_ahead_mask)
        return encode_mask, decode_mask

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


class ShiftData(TensorOp):
    def forward(self, data, state):
        target = data
        return target[:, :-1], target[:, 1:]


class MaskedCrossEntropy(LossOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def forward(self, data, state):
        y_pred, y_true = data
        mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), tf.float32)
        loss = self.loss_fn(y_true, y_pred) * mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


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


def get_estimator(data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  epochs=20,
                  em_dim=128,
                  batch_size=64,
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
        model_fn=lambda: transformer(num_layers=4,
                                     em_dim=em_dim,
                                     num_heads=8,
                                     dff=512,
                                     input_vocab=pt_tokenizer.vocab_size,
                                     target_vocab=en_tokenizer.vocab_size,
                                     max_pos_enc=1000,
                                     max_pos_dec=1000),
        optimizer_fn="adam")
    network = fe.Network(ops=[
        ShiftData(inputs="target", outputs=("target_inp", "target_real")),
        CreateMasks(inputs=("source", "target_inp"), outputs=("encode_mask", "decode_mask")),
        ModelOp(model=model, inputs=("source", "target_inp", "encode_mask", "decode_mask"), outputs="pred"),
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


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
