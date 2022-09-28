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
"""U-Net3d 3plus example."""
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import sigmoid
from tensorflow.keras.layers import BatchNormalization, Conv3D, Input, MaxPooling3D, ReLU, UpSampling3D, concatenate
from tensorflow.keras.models import Model

import fastestimator as fe
from fastestimator.dataset.data.em_3d import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Rotate, VerticalFlip
from fastestimator.op.numpyop.univariate import Minmax
from fastestimator.op.numpyop.univariate.expand_dims import ExpandDims
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.resize3d import Resize3D
from fastestimator.trace.adapt import EarlyStopping, ReduceLROnPlateau
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice

conv_config = {'activation': None, 'padding': 'same', 'kernel_initializer': 'he_normal'}


def encoderblock(x, filters, skip_fbn=False):
    if not skip_fbn:
        x = BatchNormalization(axis=4)(x)
        x = ReLU()(x)
    x = Conv3D(filters, 3, **conv_config)(x)
    x = BatchNormalization(axis=4)(x)
    x = ReLU()(x)
    x = Conv3D(filters, 3, **conv_config)(x)
    return x


def upsampleblock(x, filters, pool_size=2):
    return Conv3D(filters, 3, **conv_config)(UpSampling3D(size=(pool_size, pool_size, pool_size))(x))


def downsampleblock(x, filters, pool_size=2):
    return Conv3D(filters, 3, **conv_config)(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x))


def convblock(x, filters):
    return Conv3D(filters, 3, **conv_config)(x)


def stdconvblock(x, filters):
    x = BatchNormalization(axis=4)(x)
    x = ReLU()(x)
    x = Conv3D(filters, 3, **conv_config)(x)
    return x


def unet3d_3plus(input_size=(288, 288, 160, 1), output_classes=7, filters=64, attention=False):
    inputs = Input(input_size)
    upchannels = filters
    if attention:
        attention_upchannels = upchannels

    conv1 = encoderblock(inputs, filters, skip_fbn=True)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = encoderblock(pool1, filters * 2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = encoderblock(pool2, filters * 4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = encoderblock(pool3, filters * 8)

    up5_4 = upsampleblock(conv4, upchannels, 2)
    up5_3 = convblock(conv3, upchannels)
    down5_2 = downsampleblock(conv2, upchannels, 2)
    down5_1 = downsampleblock(conv1, upchannels, 4)

    if attention:
        up5_4 = attention_block(attention_upchannels, decoder_input=up5_4, encoder_input=up5_3)
        down5_2 = attention_block(attention_upchannels, decoder_input=down5_2, encoder_input=up5_3)
        down5_1 = attention_block(attention_upchannels, decoder_input=down5_1, encoder_input=up5_3)

    merge5 = concatenate([up5_4, up5_3, down5_2, down5_1], axis=4)
    conv5 = stdconvblock(merge5, upchannels * 4)

    up6_4 = upsampleblock(conv4, upchannels, 4)
    up6_3 = upsampleblock(conv5, upchannels, 2)
    up6_2 = convblock(conv2, upchannels)
    down6_1 = downsampleblock(conv1, upchannels, 2)

    if attention:
        up6_4 = attention_block(attention_upchannels, decoder_input=up6_4, encoder_input=up6_2)
        up6_3 = attention_block(attention_upchannels, decoder_input=up6_3, encoder_input=up6_2)
        down6_1 = attention_block(attention_upchannels, decoder_input=down6_1, encoder_input=up6_2)

    merge6 = concatenate([up6_4, up6_3, up6_2, down6_1], axis=4)
    conv6 = stdconvblock(merge6, upchannels * 4)

    up7_4 = upsampleblock(conv4, upchannels, 8)
    up7_3 = upsampleblock(conv5, upchannels, 4)
    up7_2 = upsampleblock(conv6, upchannels, 2)
    conv7_1 = convblock(conv1, upchannels)

    if attention:
        up7_4 = attention_block(attention_upchannels, decoder_input=up7_4, encoder_input=conv7_1)
        up7_3 = attention_block(attention_upchannels, decoder_input=up7_3, encoder_input=conv7_1)
        up7_2 = attention_block(attention_upchannels, decoder_input=up7_2, encoder_input=conv7_1)
    merge7 = concatenate([up7_4, up7_3, up7_2, conv7_1], axis=4)
    conv7 = stdconvblock(merge7, upchannels * 4)

    conv8 = BatchNormalization(axis=4)(conv7)
    conv8 = ReLU()(conv8)
    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=conv8)
    return model


def attention_block(n_filters: int, decoder_input, encoder_input):
    """An attention unit for Attention Unet 3d.

    Args:
        n_filters: How many filters for the convolution layer.
        decoder_input: Input tensor in the decoder section.
        encoder_input: Input tensor in the encoder section.

    Return:
        Output Keras tensor.
    """
    c1 = Conv3D(n_filters, kernel_size=1)(decoder_input)
    c1 = BatchNormalization()(c1)
    x1 = Conv3D(n_filters, kernel_size=1)(encoder_input)
    x1 = BatchNormalization()(x1)
    att = ReLU()(x1 + c1)
    att = Conv3D(1, kernel_size=1)(att)
    att = BatchNormalization()(att)
    att = sigmoid(att)
    return encoder_input * att


class ClassEncoding(NumpyOp):
    """
    One hot encode the class labels

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        no_of_classes: number of classes
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self, inputs, outputs, no_of_classes: int = 5, mode=None, ds_id=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.no_of_classes = no_of_classes

    def forward(self, data, state):
        encoded_label = np.zeros(list(data.shape) + [self.no_of_classes])
        for i in range(self.no_of_classes):
            encoded_label[:, :, :, i] = (data == i).astype(np.uint8)
        return np.uint8(encoded_label)


def get_estimator(epochs=20,
                  batch_size=1,
                  input_shape=(256, 256, 24),
                  channels=(1, ),
                  num_classes=7,
                  filters=64,
                  learning_rate=1e-3,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  log_steps=20,
                  data_dir=None):

    # step 1
    train_data, eval_data = load_data(data_dir)

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="label", mode='train')),
            Sometimes(numpy_op=VerticalFlip(image_in="image", mask_in="label", mode='train')),
            Sometimes(numpy_op=Rotate(
                image_in="image", mask_in="label", limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, mode='train')),
            ClassEncoding(inputs="label", outputs="label", no_of_classes=num_classes),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="label", outputs="label", mode='!infer'),
            ExpandDims(inputs="image", outputs="image"),
        ])

    # step 2
    model = fe.build(model_fn=lambda: unet3d_3plus(input_shape + channels, num_classes, filters),
                     optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     model_name="unet3d_3plus")

    network = fe.Network(ops=[
        Resize3D(inputs="image", outputs="image", output_shape=input_shape),
        Resize3D(inputs="label", outputs="label", output_shape=input_shape, mode='!infer'),
        ModelOp(inputs="image", model=model, outputs="pred_segment"),
        CrossEntropy(inputs=("pred_segment", "label"), outputs="ce_loss", form="binary"),
        UpdateOp(model=model, loss_name="ce_loss")
    ])

    # step 3
    traces = [
        Dice(true_key="label", pred_key="pred_segment"),
        ReduceLROnPlateau(model=model, metric="Dice", patience=7, factor=0.5, best_mode="max"),
        BestModelSaver(model=model, save_dir=save_dir, metric='Dice', save_best_mode='max'),
        EarlyStopping(monitor="Dice", compare='max', min_delta=0.005, patience=20),
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             log_steps=log_steps,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
