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
"""
This code implements LoRA training for ViT architecture in TensorFlow. All ViT encoder parameters except LoRA's weights
 are frozen. During training, the LoRA weights and the dense layer classification head are trained together.
"""
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset.data import cifar10
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


class LoRADense(layers.Layer):
    def __init__(self, lora_r, em_dim, idx):
        super().__init__()
        self.w_original = layers.Dense(em_dim)
        self.w_a = layers.Dense(lora_r, use_bias=False, name="lora_w_a_{}".format(idx))
        # the second LoRA dense initializes as 0 so that it won't affect the network prediction in the begining
        self.w_b = layers.Dense(em_dim, use_bias=False, kernel_initializer='zeros', name="lora_w_b_{}".format(idx))

    def call(self, inputs):
        x = self.w_original(inputs)
        x_lora = self.w_a(inputs)
        x_lora = self.w_b(x_lora)
        return x + x_lora


class MultiHeadAttention(layers.Layer):
    def __init__(self, lora_r, em_dim, num_heads, idx):
        super().__init__()
        assert em_dim % num_heads == 0, "model dimension must be multiple of number of heads"
        self.num_heads = num_heads
        self.em_dim = em_dim
        self.depth = em_dim // self.num_heads
        self.wq = LoRADense(lora_r, em_dim, idx)
        self.wk = layers.Dense(em_dim)
        self.wv = LoRADense(lora_r, em_dim, idx)
        self.projection = layers.Dense(em_dim)

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
        output = self.projection(concat_attention)
        return output


class EncoderLayer(layers.Layer):
    def __init__(self, lora_r, em_dim, num_heads, dff, idx, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(lora_r, em_dim, num_heads, idx)
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
    def __init__(self, lora_r, num_layers, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(lora_r, em_dim, num_heads, dff, idx=idx, rate=rate) for idx in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        return x

    def load_lora_weights(self, save_path):
        lora_vars = {var.name: var for var in self.trainable_variables if "lora_w" in var.name}
        print("Loading LoRA weights from {}...".format(save_path))
        loaded_vars = np.load(save_path)
        for var_name, var in lora_vars.items():
            var.assign(loaded_vars[var_name])

    def save_lora_weights(self, save_dir, name="lora_weight"):
        lora_vars = {var.name: var.numpy() for var in self.trainable_variables if "lora_w" in var.name}
        save_path = os.path.join(save_dir, "{}.npz".format(name))
        print("Saving LoRA weights to {}...".format(save_path))
        np.savez(save_path, **lora_vars)
        return save_path


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


def transformer_encoder_lora(image_size,
                             lora_r=4,
                             patch_size=16,
                             num_layers=12,
                             em_dim=768,
                             num_heads=12,
                             dff=3072,
                             rate=0.1):
    inputs = layers.Input(shape=image_size)
    # Patch Embedding
    x = layers.Conv2D(em_dim, kernel_size=patch_size, strides=patch_size, use_bias=False)(inputs)  #[B, H, W, em_dim]
    x = layers.Reshape((-1, em_dim))(x)  # [B, num_patches, em_dim]
    x = ClsToken(em_dim)(x)  # [B, num_patches + 1, em_dim]
    x = PositionEmbedding(image_size, patch_size, em_dim)(x)
    x = Encoder(lora_r, num_layers=num_layers, em_dim=em_dim, num_heads=num_heads, dff=dff, rate=rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])  # only need the embedding w.r.t [cls] token
    return tf.keras.Model(inputs=inputs, outputs=x)


def vision_transformer_lora(num_class,
                            image_size,
                            lora_r=4,
                            encoder_weights=None,
                            lora_weights=None,
                            patch_size=16,
                            num_layers=12,
                            em_dim=768,
                            num_heads=12,
                            dff=3072,
                            rate=0.1):
    inputs = layers.Input(shape=image_size)
    backbone = transformer_encoder_lora(image_size, lora_r, patch_size, num_layers, em_dim, num_heads, dff, rate)
    if encoder_weights:
        backbone.load_weights(encoder_weights)
    if lora_weights:
        backbone.layers[5].load_lora_weights(lora_weights)
    # freeze original encoder weights
    set_trainable_weights(backbone)
    x = backbone(inputs)
    x = layers.Dense(num_class)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    # print the number of trainable/non-trainable parameters
    num_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    num_non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables])
    print("Number of Trainable Parameters: {}".format(num_trainable_params))
    print("Number of Non-Trainable Parameters: {}".format(num_non_trainable_params))
    return model


def set_trainable_weights(model):
    # only make wa, wb weights associated with lora layer trainable
    layers = list(model._flatten_layers(include_self=False, recursive=True))
    # skip freezing any modules that are associated with lora or are parents of lora
    skipped_name_includes = [
        "model", "encoder", "encoder_layer", "multi_head_attention", "lo_ra_dense", "lora_w_a", "lora_w_b"
    ]
    for layer in layers:
        if not any(name in layer.name for name in skipped_name_includes):
            layer.trainable = False


class BestLoRASaver(BestModelSaver):
    def on_epoch_end(self, data) -> None:
        if self.monitor_op(data[self.metric], self.best):
            self.best = data[self.metric]
            self.since_best = 0
            if self.save_dir:
                self.model_path = self.model.layers[1].layers[5].save_lora_weights(self.save_dir,
                                                                                   self.model_name + "_lora")
        else:
            self.since_best += 1
        data.write_with_log(self.outputs[0], self.since_best)
        data.write_with_log(self.outputs[1], self.best)


def get_estimator(batch_size=128,
                  epochs=20,
                  save_dir=tempfile.mkdtemp(),
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  encoder_weights=None,
                  lora_weights=None):
    train_data, eval_data = cifar10.load_data()
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
    vit_lora = fe.build(
        model_fn=lambda: vision_transformer_lora(num_class=100,
                                                 image_size=(32, 32, 3),
                                                 lora_r=4,
                                                 encoder_weights=encoder_weights,
                                                 lora_weights=lora_weights,
                                                 patch_size=4,
                                                 num_layers=6,
                                                 em_dim=256,
                                                 num_heads=8,
                                                 dff=512),
        optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=vit_lora, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=vit_lora, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=vit_lora, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        BestLoRASaver(model=vit_lora, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator

if __name__ == "__main__":
    est = get_estimator()
    est.fit()