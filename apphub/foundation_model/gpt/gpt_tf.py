# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
import random
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import fastestimator as fe
from fastestimator.dataset.data import wikitext_103
from fastestimator.op.numpyop import LambdaOp, NumpyOp
from fastestimator.op.tensorop import LambdaOp as TLambdaOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver


class TextDataset(Dataset):
    def __init__(self, file_path, num_chars=5000):
        super().__init__()
        self.texts = self._read_file(file_path)
        self.num_chars = num_chars

    @staticmethod
    def _read_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def __len__(self):
        # this is just a placeholder, we use 'train_steps_per_epoch' to control training length
        return 10000

    def __getitem__(self, idx):
        start_idx = random.randint(0, len(self.texts) - self.num_chars - 1)
        random_text = self.texts[start_idx:start_idx + self.num_chars]
        return {"x": random_text[random_text.index(" ") + 1:]}  # always start from a new word


class Encode(NumpyOp):
    def __init__(self, tokenizer, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = tokenizer

    def forward(self, data, state):
        return np.array(self.tokenizer(data, truncation=True)['input_ids'])


class MultiHeadAttention(layers.Layer):
    def __init__(self, context_len, em_dim, num_heads, p_drop):
        super().__init__()
        self.num_heads = num_heads
        self.context_len = context_len
        self.em_dim = em_dim
        self.key = layers.Dense(em_dim, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
        self.query = layers.Dense(em_dim, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
        self.value = layers.Dense(em_dim, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
        self.projection = layers.Dense(em_dim, kernel_initializer=RandomNormal(stddev=0.02))
        self.dropout_attn = layers.Dropout(rate=p_drop)
        self.dropout_proj = layers.Dropout(rate=p_drop)
        self.em_dim_per_head = em_dim // num_heads

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.em_dim_per_head))  # B, seq_len, num_heads, em_dim_new
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, num_heads, seq_len, depth

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        k, q, v = self.key(x), self.query(x), self.value(x)  # B, seq, em_dim
        k = self.split_heads(k, batch_size=batch_size)  # B, num_heads, seq_len, em_dim_new
        q = self.split_heads(q, batch_size=batch_size)  # B, num_heads, seq_len, em_dim_new
        v = self.split_heads(v, batch_size=batch_size)  # B, num_heads, seq_len, em_dim_new
        # attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # B, num_heads, seq_len, seq_len
        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dimension for later normalization
        attention = matmul_qk / tf.math.sqrt(dk)
        # apply masking
        lookahead_mask = 1 - tf.linalg.band_part(tf.ones(
            (self.context_len, self.context_len)), -1, 0)  # upper tri matrix with diagonal as 0
        lookahead_mask = tf.tile(lookahead_mask[tf.newaxis, tf.newaxis, ...],
                                 [batch_size, self.num_heads, 1, 1])  # B, num_heads, seq_len, seq_len
        attention = attention - 1e9 * lookahead_mask[..., :seq_len, :seq_len]
        # softmax
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.dropout_attn(attention, training=training)
        # output reshaping
        output = tf.matmul(attention, v)  # B, num_heads, seq_len, em_dim_new
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # B, seq_len, num_heads, em_dim_new
        output = tf.reshape(output, (batch_size, -1, self.em_dim))  # B, seq_len, em_dim
        # projection
        output = self.projection(output)
        output = self.dropout_proj(output, training=training)
        return output


def point_wise_feed_forward_network(em_dim, ffwd_dim, p_drop):
    return tf.keras.Sequential([
        layers.Dense(ffwd_dim, activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.02)),  # (batch_size, seq_len, dff)
        layers.Dense(em_dim, kernel_initializer=RandomNormal(stddev=0.02)),  # (batch_size, seq_len, em_dim)
        layers.Dropout(rate=p_drop)
    ])


class AttentionBlock(layers.Layer):
    def __init__(self, context_len, em_dim, num_heads, ffwd_dim, p_drop):
        super().__init__()
        self.self_attention = MultiHeadAttention(context_len, em_dim, num_heads, p_drop)
        self.ffwd = point_wise_feed_forward_network(em_dim, ffwd_dim, p_drop)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x = x + self.self_attention(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x


class GPT(tf.keras.Model):
    def __init__(self, num_blocks, vocab_size, context_len, em_dim, num_heads, ffwd_dim, p_drop=0.2):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, em_dim, embeddings_initializer=RandomNormal(stddev=0.02))
        self.position_embedding = layers.Embedding(context_len,
                                                   em_dim,
                                                   embeddings_initializer=RandomNormal(stddev=0.02))
        self.blocks = tf.keras.Sequential(
            [AttentionBlock(context_len, em_dim, num_heads, ffwd_dim, p_drop) for _ in range(num_blocks)])
        self.final_norm = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(vocab_size, kernel_initializer=RandomNormal(stddev=0.02))
        self.pos_idx = tf.convert_to_tensor(list(range(context_len)))
        self.build(input_shape=(None, None))

    def call(self, x):
        seq_len = tf.shape(x)[1]
        token_em = self.token_embedding(x)
        position_em = self.position_embedding(self.pos_idx[None, :seq_len])
        x = token_em + position_em
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


def generate_response(prompt, model, tokenizer, max_response_token=128, context_len=512):
    tokens = tf.convert_to_tensor(tokenizer(prompt, truncation=True)['input_ids'])
    assert tokens.shape[-1] <= context_len, "prompt exceeding maximum input tokens"
    tokens = tokens[tf.newaxis, ...]  # add batch dimension
    responses = []
    for _ in range(max_response_token):
        input_tokens = tokens[:, -context_len:]
        # get prediction
        logits = model(input_tokens, training=False)  # [B, seq, vocab_size]
        # focus only the last time step
        logits = logits[:, -1, :]
        probs = tf.math.softmax(logits, axis=-1)
        probs = tf.cast(probs, tf.float64)
        probs = probs / tf.reduce_sum(probs)  # to remove numerical inconsistency
        idx_next = np.argmax(np.random.multinomial(n=1, pvals=probs[0].numpy()))
        responses.append(idx_next)
        tokens = tf.concat([tokens, tf.convert_to_tensor([[idx_next]])], axis=-1)
    responses = tokenizer.decode(responses)
    return responses


def get_estimator(data_dir=None,
                  epochs=50,
                  batch_size=32,
                  context_len=512,
                  num_blocks=6,
                  em_dim=1024,
                  ffwd_dim=4096,
                  num_heads=16,
                  save_dir=tempfile.mkdtemp(),
                  train_steps_per_epoch=3000,
                  eval_steps_per_epoch=500):
    # first load the data
    train_data, eval_data, test_data = wikitext_103.load_data(data_dir)
    train_data, eval_data, test_data = TextDataset(train_data), TextDataset(eval_data), TextDataset(test_data)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Encode(inputs="x", outputs="x", tokenizer=tokenizer),
            LambdaOp(fn=lambda x: x[:context_len + 1], inputs="x", outputs="x")  # get 1 more token for target
        ])
    model = fe.build(
        model_fn=lambda: GPT(num_blocks=num_blocks,
                             vocab_size=tokenizer.vocab_size,
                             context_len=context_len,
                             em_dim=em_dim,
                             num_heads=num_heads,
                             ffwd_dim=ffwd_dim,
                             p_drop=0.3),
        optimizer_fn=lambda: tf.optimizers.Adam(3e-4))
    network = fe.Network(ops=[
        TLambdaOp(fn=lambda x: (x[..., :-1], x[..., 1:]), inputs="x", outputs=("input", "target")),
        ModelOp(model=model, inputs="input", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "target"), outputs="ce", form='sparse', from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=BestModelSaver(model=model, save_dir=save_dir),
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
