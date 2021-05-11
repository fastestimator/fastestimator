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
The FastEstimator implementation of SimCLR with ResNet9 on CIFAIR10.
This code took reference from google implementation (https://github.com/google-research/simclr).
Note that we use the ciFAIR10 dataset instead (https://cvjena.github.io/cifair/)
"""
import tempfile

import tensorflow as tf
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset.data.cifair10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ColorJitter, GaussianBlur, Onehot, ToFloat, ToGray
from fastestimator.op.tensorop import LambdaOp, TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver, ModelSaver
from fastestimator.trace.metric import Accuracy


def ResNet9(input_size=(32, 32, 3), head_len=128, classes=10):
    """A small 9-layer ResNet Tensorflow model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow ResNet9 model.
    """

    # prep layers
    inp = layers.Input(shape=input_size)
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer1
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    # layer2
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer3
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    # layers4
    x = layers.GlobalMaxPool2D()(x)
    code = layers.Flatten()(x)

    p_head = layers.Dense(head_len)(code)
    model_con = tf.keras.Model(inputs=inp, outputs=p_head)

    s_head = layers.Dense(classes)(code)
    s_head = layers.Activation('softmax', dtype='float32')(s_head)
    model_finetune = tf.keras.Model(inputs=inp, outputs=s_head)

    return model_con, model_finetune


def residual(x, num_channel):
    """A ResNet unit for ResNet9.

    Args:
        x: Input Keras tensor.
        num_channel: The number of layer channel.

    Return:
        Output Keras tensor.
    """
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


class NTXentOp(TensorOp):
    def __init__(self, arg1, arg2, outputs, temperature=1.0, mode=None):
        super().__init__(inputs=(arg1, arg2), outputs=outputs, mode=mode)
        self.temperature = temperature

    def forward(self, data, state):
        arg1, arg2 = data
        loss = NTXent(arg1, arg2, self.temperature)
        return loss


def NTXent(A, B, temperature):
    large_number = 1e9
    batch_size = tf.shape(A)[0]
    A = tf.math.l2_normalize(A, -1)
    B = tf.math.l2_normalize(B, -1)

    mask = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), 2 * batch_size)

    aa = tf.matmul(A, A, transpose_b=True) / temperature
    aa = aa - mask * large_number
    ab = tf.matmul(A, B, transpose_b=True) / temperature
    bb = tf.matmul(B, B, transpose_b=True) / temperature
    bb = bb - mask * large_number
    ba = tf.matmul(B, A, transpose_b=True) / temperature
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([ab, aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([ba, bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss, ab, labels


def pretrain_model(epochs, batch_size, max_train_steps_per_epoch, save_dir):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x"),

            # augmentation 1
            RandomCrop(32, 32, image_in="x", image_out="x_aug"),
            Sometimes(HorizontalFlip(image_in="x_aug", image_out="x_aug"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug", outputs="x_aug", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug", outputs="x_aug"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug", outputs="x_aug", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ToFloat(inputs="x_aug", outputs="x_aug"),

            # augmentation 2
            RandomCrop(32, 32, image_in="x", image_out="x_aug2"),
            Sometimes(HorizontalFlip(image_in="x_aug2", image_out="x_aug2"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug2", outputs="x_aug2", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug2", outputs="x_aug2"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug2", outputs="x_aug2", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ToFloat(inputs="x_aug2", outputs="x_aug2")
        ])

    # step 2: prepare network
    model_con, model_finetune = fe.build(model_fn=ResNet9, optimizer_fn=["adam", "adam"])
    network = fe.Network(ops=[
        LambdaOp(lambda x, y: tf.concat([x, y], axis=0), inputs=["x_aug", "x_aug2"], outputs="x_com"),
        ModelOp(model=model_con, inputs="x_com", outputs="y_com"),
        LambdaOp(lambda x: tf.split(x, 2, axis=0), inputs="y_com", outputs=["y_pred", "y_pred2"]),
        NTXentOp(arg1="y_pred", arg2="y_pred2", outputs=["NTXent", "logit", "label"]),
        UpdateOp(model=model_con, loss_name="NTXent")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="label", pred_key="logit", mode="train", output_name="contrastive_accuracy"),
        ModelSaver(model=model_con, save_dir=save_dir),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             monitor_names="contrastive_accuracy")
    estimator.fit()

    return model_con, model_finetune


def finetune_model(model, epochs, batch_size, max_train_steps_per_epoch, save_dir):
    train_data, test_data = load_data()
    train_data = train_data.split(0.1)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=test_data,
                           batch_size=batch_size,
                           ops=[
                               ToFloat(inputs="x", outputs="x"),
                           ])

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=["y_pred", "y"], outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    estimator.fit()


def fastestimator_run(epochs_pretrain=50,
                      epochs_finetune=10,
                      batch_size=512,
                      max_train_steps_per_epoch=None,
                      save_dir=tempfile.mkdtemp()):

    model_con, model_finetune = pretrain_model(epochs_pretrain, batch_size, max_train_steps_per_epoch, save_dir)
    finetune_model(model_finetune, epochs_finetune, batch_size, max_train_steps_per_epoch, save_dir)


if __name__ == "__main__":
    fastestimator_run()
