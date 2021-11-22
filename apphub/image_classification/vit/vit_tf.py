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
"""
Vision Transformer TensorFlow Implementation
"""
import tempfile

import tensorflow as tf
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset.data import cifair10, cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
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
        assert em_dim % num_heads == 0, "model dimension must be multiple of number of heads"
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

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # B, seq_len, em_dim
        k = self.wk(k)  # B, seq_len, em_dim
        v = self.wv(v)  # B, seq_len, em_dim
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v)
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

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        return x


class PositionEmbedding(layers.Layer):
    def __init__(self, image_size, patch_size, em_dim):
        super().__init__()
        h, w, _ = image_size
        assert h % patch_size == 0 and w % patch_size == 0, "image size must be an integer multiple of patch size"
        self.position_embedding = tf.Variable(tf.zeros(shape=(1, h * w // patch_size**2 + 1, em_dim)),
                                              trainable=True,
                                              name="position_embedding")

    def call(self, x):
        return x + self.position_embedding


class ClsToken(layers.Layer):
    def __init__(self, em_dim):
        super().__init__()
        self.cls_token = tf.Variable(tf.zeros(shape=(1, 1, em_dim)), trainable=True, name="cls_token")
        self.em_dim = em_dim

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.concat([tf.broadcast_to(self.cls_token, (batch_size, 1, self.em_dim)), x], axis=1)


def transformer_encoder(image_size, patch_size=16, num_layers=12, em_dim=768, num_heads=12, dff=3072, rate=0.1):
    inputs = layers.Input(shape=image_size)
    # Patch Embedding
    x = layers.Conv2D(em_dim, kernel_size=patch_size, strides=patch_size, use_bias=False)(inputs)  #[B, H, W, em_dim]
    x = layers.Reshape((-1, em_dim))(x)  # [B, num_patches, em_dim]
    x = ClsToken(em_dim)(x)  # [B, num_patches + 1, em_dim]
    x = PositionEmbedding(image_size, patch_size, em_dim)(x)
    x = Encoder(num_layers=num_layers, em_dim=em_dim, num_heads=num_heads, dff=dff, rate=rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])  # only need the embedding w.r.t [cls] token
    return tf.keras.Model(inputs=inputs, outputs=x)


def vision_transformer(num_class,
                       image_size,
                       weights_path=None,
                       patch_size=16,
                       num_layers=12,
                       em_dim=768,
                       num_heads=12,
                       dff=3072,
                       rate=0.1):
    inputs = layers.Input(shape=image_size)
    backbone = transformer_encoder(image_size, patch_size, num_layers, em_dim, num_heads, dff, rate)
    if weights_path:
        backbone.load_weights(weights_path)
    x = backbone(inputs)
    x = layers.Dense(num_class)(x)
    return backbone, tf.keras.Model(inputs=inputs, outputs=x)


def pretrain(batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             max_train_steps_per_epoch=None,
             max_eval_steps_per_epoch=None):
    train_data, eval_data = cifair100.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1)
        ])
    backbone, vit = fe.build(
        model_fn=lambda: vision_transformer(
            num_class=100, image_size=(32, 32, 3), patch_size=4, num_layers=6, em_dim=256, num_heads=8, dff=512),
        optimizer_fn=[None, lambda: tf.optimizers.SGD(0.01, momentum=0.9)])
    network = fe.Network(ops=[
        ModelOp(model=vit, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=vit, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=backbone, save_dir=model_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch)
    estimator.fit(warmup=False)
    return traces[1].model_path  # return the weights path


def finetune(weights_path,
             batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             max_train_steps_per_epoch=None,
             max_eval_steps_per_epoch=None):
    train_data, eval_data = cifair10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1)
        ])
    _, model = fe.build(
        model_fn=lambda: vision_transformer(num_class=10,
                                            weights_path=weights_path,
                                            image_size=(32, 32, 3),
                                            patch_size=4,
                                            num_layers=6,
                                            em_dim=256,
                                            num_heads=8,
                                            dff=512),
        optimizer_fn=[None, lambda: tf.optimizers.SGD(0.01, momentum=0.9)])
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch)
    estimator.fit(warmup=False)


def fastestimator_run(batch_size=128,
                      pretrain_epochs=100,
                      finetune_epochs=1,
                      max_train_steps_per_epoch=None,
                      max_eval_steps_per_epoch=None):
    weights_path = pretrain(batch_size=batch_size,
                            epochs=pretrain_epochs,
                            max_train_steps_per_epoch=max_train_steps_per_epoch,
                            max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    finetune(weights_path,
             batch_size=batch_size,
             epochs=finetune_epochs,
             max_train_steps_per_epoch=max_train_steps_per_epoch,
             max_eval_steps_per_epoch=max_eval_steps_per_epoch)


if __name__ == "__main__":
    fastestimator_run()
