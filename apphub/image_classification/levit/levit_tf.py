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
LeVIT TensorFlow Implementation
"""
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset.data import cifair10, cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

specification = {
    'LeViT_128S': {
        'embed_dim': [128, 256, 384], 'key_dim': 16, 'num_heads': [4, 6, 8], 'depth': [2, 3, 4], 'drop_path': 0
    },
    'LeViT_256': {
        'embed_dim': [256, 384, 512], 'key_dim': 32, 'num_heads': [4, 6, 8], 'depth': [4, 4, 4], 'drop_path': 0
    },
    'LeViT_384': {
        'embed_dim': [384, 512, 768], 'key_dim': 32, 'num_heads': [6, 9, 12], 'depth': [4, 4, 4], 'drop_path': 0.1
    },
}


def hard_swish(features):
    """Computes a hard version of the swish function.

    This operation can be used to reduce computational cost and improve
    quantization for edge devices.

    Args:
        features: A `Tensor` representing preactivation values.

    Returns:
        The activation value.
    """
    return features * tf.nn.relu6(features + tf.cast(3., features.dtype)) * (1. / 6.)


class Backbone(layers.Layer):
    def __init__(self, out_channels):
        super(Backbone, self).__init__()
        self.convolution_layer1 = layers.Conv2D(filters=out_channels // 8,
                                                kernel_size=3,
                                                strides=2,
                                                padding="same",
                                                use_bias=False)
        self.batch_norm1 = layers.BatchNormalization(gamma_initializer='ones')
        self.convolution_layer2 = layers.Conv2D(filters=out_channels // 4,
                                                kernel_size=3,
                                                strides=2,
                                                padding="same",
                                                use_bias=False)
        self.batch_norm2 = layers.BatchNormalization(gamma_initializer='ones')
        self.convolution_layer3 = layers.Conv2D(filters=out_channels // 2,
                                                kernel_size=3,
                                                strides=2,
                                                padding="same",
                                                use_bias=False)
        self.batch_norm3 = layers.BatchNormalization(gamma_initializer='ones')
        self.convolution_layer4 = layers.Conv2D(filters=out_channels,
                                                kernel_size=3,
                                                strides=2,
                                                padding="same",
                                                use_bias=False)
        self.batch_norm4 = layers.BatchNormalization(gamma_initializer='ones')

    def call(self, x):
        x = hard_swish(self.batch_norm1(self.convolution_layer1(x)))
        x = hard_swish(self.batch_norm2(self.convolution_layer2(x)))
        x = hard_swish(self.batch_norm3(self.convolution_layer3(x)))
        x = hard_swish(self.batch_norm4(self.convolution_layer4(x)))
        return x


class Residual(layers.Layer):
    def __init__(self, module, drop_rate=0.):
        super(Residual, self).__init__()
        self.module = module
        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, training):
        return x + self.dropout(self.module(x), training=training)


class LinearNorm(layers.Layer):
    def __init__(self, out_channels, bn_weight_init=1):
        super(LinearNorm, self).__init__()
        self.batch_norm = layers.BatchNormalization(gamma_initializer=tf.constant_initializer(bn_weight_init))
        self.linear = layers.Dense(out_channels, activation=None)

    def call(self, x):
        x = self.linear(x)
        shape = x.get_shape().as_list()
        x = tf.reshape(self.batch_norm(tf.reshape(x, (-1, shape[2]))), shape)
        return x


class Downsample(layers.Layer):
    def __init__(self, stride, resolution):
        super(Downsample, self).__init__()
        self.stride = stride
        self.resolution = resolution

    def call(self, x):
        batch_size, _, channels = x.get_shape().as_list()
        x = tf.reshape(x, (batch_size, self.resolution, self.resolution, channels))
        x = x[:, ::self.stride, ::self.stride]
        return tf.reshape(x, (batch_size, -1, channels))


class NormLinear(layers.Layer):
    def __init__(self, out_channels, bias=True, std=0.02, drop=0.0):
        super(NormLinear, self).__init__()
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(drop)
        self.linear = layers.Dense(out_channels,
                                   activation=None,
                                   use_bias=bias,
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=std))

    def call(self, x, training):
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.linear(x)
        return x


class MLP(layers.Layer):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.linear_up = LinearNorm(hidden_dim)
        self.linear_down = LinearNorm(input_dim)

    def call(self, x):
        return self.linear_down(hard_swish(self.linear_up(x)))


class Attention(layers.Layer):
    def __init__(self, input_dim, key_dim, num_attention_heads=8, attention_ratio=4, resolution=14):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio

        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads

        self.queries_keys_values = LinearNorm(self.out_dim_keys_values)
        self.projection = LinearNorm(input_dim)

    def call(self, x):
        batch_size, seq_length, _ = x.get_shape().as_list()

        queries_keys_values = self.queries_keys_values(x)

        query, key, value = tf.split(tf.reshape(queries_keys_values, (batch_size, seq_length, self.num_attention_heads, -1)), [
                self.key_dim, self.key_dim, self.attention_ratio * self.key_dim],axis=3)

        query = tf.transpose(query, (0, 2, 1, 3))
        key = tf.transpose(key, (0, 2, 1, 3))
        value = tf.transpose(value, (0, 2, 1, 3))
        attention = tf.matmul(query, key, transpose_b=True) * self.scale
        attention = tf.nn.softmax(attention, axis=-1)
        hidden_state = tf.reshape(tf.transpose(tf.matmul(attention, value), (0, 1, 3, 2)),
                                  (batch_size, seq_length, self.out_dim_projection))
        hidden_state = self.projection(hard_swish(hidden_state))
        return hidden_state


class AttentionDownsample(layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 key_dim,
                 num_attention_heads,
                 attention_ratio,
                 stride,
                 resolution_in,
                 resolution_out):
        super(AttentionDownsample, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.resolution_out = resolution_out
        self.resolution_in = resolution_in
        # resolution_in is the intial resolution, resoloution_out is final resolution after downsampling
        self.keys_values = LinearNorm(self.out_dim_keys_values)
        self.queries_subsample = Downsample(stride, resolution_in)
        self.queries = LinearNorm(key_dim * num_attention_heads)
        self.projection = LinearNorm(output_dim)

    def call(self, x):
        batch_size, seq_length, _ = x.get_shape().as_list()

        key, value = tf.split(tf.reshape(self.keys_values(x), (
            batch_size, seq_length, self.num_attention_heads,
            -1)), [self.key_dim, self.attention_ratio * self.key_dim],
                      axis=3)

        key = tf.transpose(key, (0, 2, 1, 3))
        value = tf.transpose(value, (0, 2, 1, 3))

        query = self.queries(self.queries_subsample(x))
        query = tf.transpose(
            tf.reshape(query, (batch_size, self.resolution_out**2, self.num_attention_heads, self.key_dim)),
            (0, 2, 1, 3))

        attention = tf.matmul(query, key, transpose_b=True) * self.scale
        attention = tf.nn.softmax(attention, axis=-1)
        x = tf.reshape(tf.transpose(tf.matmul(attention, value), (0, 1, 3, 2)),
                       (batch_size, -1, self.out_dim_projection))
        x = self.projection(hard_swish(x))
        return x


def levit_stage(embed_dim,
                key_dim,
                num_attention_heads,
                resolution,
                depth,
                attention_ratio,
                mlp_ratio,
                drop_path):
    stages = []
    for i in range(depth):
        stages.append(
            Residual(
                Attention(input_dim=embed_dim,
                          key_dim=key_dim,
                          num_attention_heads=num_attention_heads,
                          attention_ratio=attention_ratio,
                          resolution=resolution),
                drop_path))

        if mlp_ratio > 0:
            h = int(embed_dim * mlp_ratio)
            stages.append(
                Residual(MLP(input_dim=embed_dim, hidden_dim=h),
                         drop_path))
    return tf.keras.Sequential(stages)


def levit_downsample(input_dim, output_dim, resolution, resolution_out, down_ops, drop_path):
    stages = []
    stages.append(
        AttentionDownsample(input_dim=input_dim,
                            output_dim=output_dim,
                            key_dim=down_ops['key_dim'],
                            num_attention_heads=down_ops['num_heads'],
                            attention_ratio=down_ops['attn_ratio'],
                            stride=down_ops['stride'],
                            resolution_in=resolution,
                            resolution_out=resolution_out))
    if down_ops['mlp_ratio'] > 0:  # mlp_ratio
        h = int(output_dim * down_ops['mlp_ratio'])
        stages.append(
            Residual(MLP(input_dim=output_dim, hidden_dim=h), drop_path))
    return tf.keras.Sequential(stages)


class LeVIT(tf.keras.Model):
    def __init__(self,
                 image_dim,
                 patch_size,
                 num_classes,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attention_ratio=[2],
                 mlp_ratio=[2],
                 down_ops={},
                 distillation=False,
                 drop_path=0.):
        super(LeVIT, self).__init__()
        self.distillation = distillation
        input_resolution_stage1 = image_dim // patch_size
        input_resolution_stage2 = (input_resolution_stage1 - 1) // down_ops[1]['stride'] + 1
        input_resolution_stage3 = (input_resolution_stage2 - 1) // down_ops[2]['stride'] + 1

        self.backbone = Backbone(embed_dim[0])

        self.stage1 = levit_stage(embed_dim=embed_dim[0],
                                  key_dim=key_dim[0],
                                  num_attention_heads=num_heads[0],
                                  resolution=input_resolution_stage1,
                                  depth=depth[0],
                                  attention_ratio=attention_ratio[0],
                                  mlp_ratio=mlp_ratio[0],
                                  drop_path=drop_path)

        self.stage1_downsample = levit_downsample(input_dim=embed_dim[0],
                                                  output_dim=embed_dim[1],
                                                  resolution=input_resolution_stage1,
                                                  resolution_out=input_resolution_stage2,
                                                  down_ops=down_ops[1],
                                                  drop_path=drop_path)

        self.stage2 = levit_stage(embed_dim=embed_dim[1],
                                  key_dim=key_dim[1],
                                  num_attention_heads=num_heads[1],
                                  resolution=input_resolution_stage2,
                                  depth=depth[1],
                                  attention_ratio=attention_ratio[1],
                                  mlp_ratio=mlp_ratio[1],
                                  drop_path=drop_path)

        self.stage2_downsample = levit_downsample(input_dim=embed_dim[1],
                                                  output_dim=embed_dim[2],
                                                  resolution=input_resolution_stage2,
                                                  resolution_out=input_resolution_stage3,
                                                  down_ops=down_ops[2],
                                                  drop_path=drop_path)

        self.stage3 = levit_stage(embed_dim=embed_dim[2],
                                  key_dim=key_dim[2],
                                  num_attention_heads=num_heads[2],
                                  resolution=input_resolution_stage3,
                                  depth=depth[2],
                                  attention_ratio=attention_ratio[2],
                                  mlp_ratio=mlp_ratio[2],
                                  drop_path=drop_path)

        self.class_head = NormLinear(num_classes) if num_classes > 0 else layers.Identity()

        if self.distillation:
            self.distll_class_head = NormLinear(num_classes) if num_classes > 0 else layers.Identity()

    def call(self, x, training):
        x = self.backbone(x)
        batch_size, _, _, channels = x.get_shape().as_list()
        x = tf.reshape(x, (batch_size, -1, channels))
        x = self.stage1(x)
        x = self.stage1_downsample(x)
        x = self.stage2(x)
        x = self.stage2_downsample(x)
        x = self.stage3(x)
        x = tf.math.reduce_mean(x, -1)

        if self.distillation:
            x = self.class_head(x), self.distll_class_head(x)
            if not training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.class_head(x)

        return x


def LeViT_128S(image_dim, num_classes=1000, distillation=False, pretrained=False):
    return model_factory(image_dim=image_dim,
                         **specification['LeViT_128S'],
                         num_classes=num_classes,
                         distillation=distillation)


def LeViT_256(image_dim, num_classes=1000, distillation=False, pretrained=False):
    return model_factory(image_dim=image_dim,
                         **specification['LeViT_256'],
                         num_classes=num_classes,
                         distillation=distillation)


def LeViT_384(image_dim, num_classes=1000, distillation=False, pretrained=False):
    return model_factory(image_dim=image_dim,
                         **specification['LeViT_384'],
                         num_classes=num_classes,
                         distillation=distillation)


def model_factory(image_dim,
                  embed_dim,
                  key_dim,
                  depth,
                  num_heads,
                  drop_path,
                  num_classes,
                  distillation):
    model = LeVIT(
        image_dim,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[key_dim] * 3,
        depth=depth,
        attention_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops={
            1: {
                'key_dim': key_dim, 'num_heads': embed_dim[0] // key_dim, 'attn_ratio': 4, 'mlp_ratio': 2, 'stride': 2
            },
            2: {
                'key_dim': key_dim, 'num_heads': embed_dim[1] // key_dim, 'attn_ratio': 4, 'mlp_ratio': 2, 'stride': 2
            },
        },
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation)

    return model


def pretrain(batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             train_steps_per_epoch=None,
             eval_steps_per_epoch=None,
             log_steps=100):

    train_data, eval_data = cifair100.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Resize(image_in="x", image_out="x", height=224, width=224),
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            Onehot(inputs="y", outputs="y", mode="train", num_classes=100, label_smoothing=0.05)
        ])

    # training it from scratch to demonstrate custom pretraining and fine tuning.
    levit_model = LeViT_256(image_dim=224, num_classes=100, pretrained=False)
    levit_model.build(input_shape=(batch_size, 224, 224, 3))

    model = fe.build(model_fn=lambda: levit_model, optimizer_fn="adam")

    def lr_schedule_warmup(step, train_steps_epoch, init_lr):
        warmup_steps = train_steps_epoch * 3
        if step < warmup_steps:
            lr = init_lr / warmup_steps * step
        else:
            lr = init_lr
        return lr

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce", mode="train")
    ])

    init_lr = 1e-2 / 64 * batch_size

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max"),
    ]

    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_schedule_warmup(
                step, train_steps_epoch=np.ceil(len(train_data) / batch_size), init_lr=init_lr)),
        4:
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(
                epoch, cycle_length=epochs - 3, init_lr=init_lr, min_lr=init_lr / 100, start=4))
    }

    traces.append(EpochScheduler(lr_schedule))

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch,
                             log_steps=log_steps)

    estimator.fit(warmup=False)
    return model


def finetune(pretrained_model,
             batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             train_steps_per_epoch=None,
             eval_steps_per_epoch=None,
             log_steps=100):

    train_data, eval_data = cifair10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Resize(image_in="x", image_out="x", height=224, width=224),
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.05)
        ])
    levit_model = LeViT_256(image_dim=224, num_classes=10)
    levit_model.build(input_shape=(batch_size, 224, 224, 3))

    model = fe.build(model_fn=lambda: levit_model, optimizer_fn="adam")

    # load the encoder's weigh
    for i in range(len(pretrained_model.layers) - 1):
        model.layers[i].set_weights(model.layers[i].weights)

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
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch,
                             log_steps=log_steps)
    estimator.fit(warmup=False)


def fastestimator_run(batch_size=64,
                      pretrain_epochs=100,
                      finetune_epochs=1,
                      model_dir=tempfile.mkdtemp(),
                      train_steps_per_epoch=None,
                      eval_steps_per_epoch=None):

    pretrained_model = pretrain(batch_size=batch_size,
                                epochs=pretrain_epochs,
                                model_dir=model_dir,
                                train_steps_per_epoch=train_steps_per_epoch,
                                eval_steps_per_epoch=eval_steps_per_epoch)
    finetune(pretrained_model,
             batch_size=batch_size,
             epochs=finetune_epochs,
             model_dir=model_dir,
             train_steps_per_epoch=train_steps_per_epoch,
             eval_steps_per_epoch=eval_steps_per_epoch)


if __name__ == "__main__":
    fastestimator_run()
