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
"""U-Net lung segmentation example.
"""


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

from fastestimator.dataset import montgomery
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Dice
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.processing import Minmax, Reshape
from fastestimator.record.preprocess import ImageReader, Resize
from fastestimator.record.record import RecordWriter

def unet(input_name, output_name, input_shape=(256, 256, 1)):
    conv_config = {'activation':'relu', 'padding':'same', 'kernel_initializer':'he_normal'}

    inputs = Input(shape=input_shape, name=input_name)

    conv1 = Conv2D(32, (3, 3), **conv_config)(inputs)
    conv1 = Conv2D(32, (3, 3), **conv_config)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), **conv_config)(pool1)
    conv2 = Conv2D(64, (3, 3), **conv_config)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), **conv_config)(pool2)
    conv3 = Conv2D(128, (3, 3), **conv_config)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), **conv_config)(pool3)
    conv4 = Conv2D(256, (3, 3), **conv_config)(conv4)
    #     conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), **conv_config)(pool4)
    conv5 = Conv2D(512, (3, 3), **conv_config)(conv5)
    #     drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(256 ,(2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), **conv_config)(up6)
    conv6 = Conv2D(256, (3, 3), **conv_config)(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), **conv_config)(up7)
    conv7 = Conv2D(128, (3, 3), **conv_config)(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), **conv_config)(up8)
    conv8 = Conv2D(64, (3, 3), **conv_config)(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), **conv_config)(up9)
    conv9 = Conv2D(32, (3, 3), **conv_config)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name=output_name)(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


class BinaryCrossentropy(Loss):
    def calculate_loss(self, batch, state):
        y_pred = batch['pred_segment']
        y_true = batch['mask']
        return tf.losses.BinaryCrossentropy()(y_true,y_pred)

def get_estimator():

    train_csv_path, eval_cvs_path, path = montgomery.load_and_set_data()
    writer = RecordWriter(
        train_data=train_csv_path, validation_data=eval_cvs_path, ops=[
            ImageReader(grey_scale=True,inputs="imgpath", parent_path=path, outputs="image"),
            ImageReader(grey_scale=True,inputs="mask", parent_path=path, outputs="mask"),
            Resize(inputs="image", target_size=(512, 512), outputs="image"),
            Resize(inputs="mask", target_size=(512, 512), outputs="mask"),
        ])

    pipeline = Pipeline(batch_size=8, data=writer, ops=[Minmax(inputs="image", outputs="image"),
                                                        Minmax(inputs="mask", outputs="mask"),
                                                        Reshape(shape=(512, 512, 1), inputs="image", outputs="image"),
                                                        Reshape(shape=(512, 512, 1), inputs="mask", outputs="mask")])


    model = build(keras_model=unet("imgpath", "mask", input_shape=(512, 512, 1)), loss=BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    network = Network(ops=[ModelOp(inputs="image", model=model, outputs="pred_segment")])

    estimator = Estimator(network=network, pipeline=pipeline, epochs=25, log_steps=20,
                          traces=[Dice("mask", "pred_segment")])

    return estimator
# command to run this script: fastestimator train lung_segmentation.py --inputs /tmp/FE_MONTGOMERY/FE
