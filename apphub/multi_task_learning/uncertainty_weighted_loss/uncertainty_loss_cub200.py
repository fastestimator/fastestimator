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
import os
import tempfile

import numpy as np

import fastestimator as fe
import tensorflow as tf
from fastestimator import RecordWriter
from fastestimator.architecture.uncertaintyloss import UncertaintyLoss
from fastestimator.architecture.unet import UNet
from fastestimator.dataset import cub200
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ImageReader, MatReader, Reshape, Resize
from fastestimator.op.tensorop import Augmentation2D, BinaryCrossentropy, Loss, ModelOp, Rescale, \
    SparseCategoricalCrossentropy
from fastestimator.trace import Accuracy, Dice, ModelSaver
from tensorflow.keras import layers, models


def ResUnet50(input_shape=(512, 512, 3), num_classes=200):
    inputs = layers.Input(shape=input_shape)
    resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling=None, input_tensor=inputs)
    assert resnet50.layers[4].name == "conv1_relu"
    C1 = resnet50.layers[4].output  # 256 x 256 x 64
    assert resnet50.layers[38].name == "conv2_block3_out"
    C2 = resnet50.layers[38].output  # 128 x 128 x 256
    assert resnet50.layers[80].name == "conv3_block4_out"
    C3 = resnet50.layers[80].output  # 64 x 64 x 512
    assert resnet50.layers[142].name == "conv4_block6_out"
    C4 = resnet50.layers[142].output  # 32 x 32 x 1024
    assert resnet50.layers[-1].name == "conv5_block3_out"
    C5 = resnet50.layers[-1].output  # 16 x 16 x 2048
    #classification subnet
    label = layers.GlobalMaxPool2D()(C5)
    label = layers.Flatten()(label)
    label = layers.Dense(num_classes, activation='softmax')(label)
    #segmentation subnet
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    up6 = layers.Conv2D(512, 3, **conv_config)(layers.UpSampling2D(size=(2, 2))(C5))  # 32 x 32 x 512
    merge6 = layers.concatenate([C4, up6], axis=3)  # 32 x 32 x 1536
    conv6 = layers.Conv2D(512, 3, **conv_config)(merge6)  # 32 x 32 x 512
    conv6 = layers.Conv2D(512, 3, **conv_config)(conv6)  # 32 x 32 x 512
    up7 = layers.Conv2D(256, 3, **conv_config)(layers.UpSampling2D(size=(2, 2))(conv6))  # 64 x 64 x 256
    merge7 = layers.concatenate([C3, up7], axis=3)  # 64 x 64 x 768
    conv7 = layers.Conv2D(256, 3, **conv_config)(merge7)  # 64 x 64 x 256
    conv7 = layers.Conv2D(256, 3, **conv_config)(conv7)  # 64 x 64 x 256
    up8 = layers.Conv2D(128, 3, **conv_config)(layers.UpSampling2D(size=(2, 2))(conv7))  # 128 x 128 x 128
    merge8 = layers.concatenate([C2, up8], axis=3)  # 128 x 128 x 384
    conv8 = layers.Conv2D(128, 3, **conv_config)(merge8)  # 128 x 128 x 128
    conv8 = layers.Conv2D(128, 3, **conv_config)(conv8)  # 128 x 128 x 128
    up9 = layers.Conv2D(64, 3, **conv_config)(layers.UpSampling2D(size=(2, 2))(conv8))  # 256 x 256 x 64
    merge9 = layers.concatenate([C1, up9], axis=3)  # 256 x 256 x 128
    conv9 = layers.Conv2D(64, 3, **conv_config)(merge9)  # 256 x 256 x 64
    conv9 = layers.Conv2D(64, 3, **conv_config)(conv9)  # 256 x 256 x 64
    up10 = layers.Conv2D(2, 3, **conv_config)(layers.UpSampling2D(size=(2, 2))(conv9))  # 512 x 512 x 2
    mask = layers.Conv2D(1, 1, activation='sigmoid')(up10)
    model = tf.keras.Model(inputs=inputs, outputs=[label, mask])
    return model


class SelectDictKey(NumpyOp):
    def forward(self, data, state):
        data = data['seg']
        return data


def get_estimator(batch_size=8, epochs=25, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
    # load CUB200 dataset.
    csv_path, path = cub200.load_data("/data/data")
    writer = RecordWriter(
        save_dir=os.path.join(path, "tfrecords"),
        train_data=csv_path,
        validation_data=0.2,
        ops=[
            ImageReader(inputs='image', parent_path=path),
            Resize(target_size=(512, 512), keep_ratio=True, outputs='image'),
            MatReader(inputs='annotation', parent_path=path),
            SelectDictKey(),
            Resize((512, 512), keep_ratio=True),
            Reshape(shape=(512, 512, 1), outputs="annotation")
        ])
    # data pipeline
    pipeline = fe.Pipeline(
        batch_size=batch_size,
        data=writer,
        ops=[
            Augmentation2D(inputs=("image", "annotation"),
                           outputs=("image", "annotation"),
                           mode="train",
                           rotation_range=15.0,
                           zoom_range=[0.8, 1.2],
                           flip_left_right=True),
            Rescale(inputs='image', outputs='image')
        ])

    # Network
    opt = tf.optimizers.Adam(learning_rate=0.0001)
    resunet50 = fe.build(model_def=ResUnet50, model_name="resunet50", optimizer=opt, loss_name="total_loss")
    uncertainty = fe.build(model_def=UncertaintyLoss, model_name="uncertainty", optimizer=opt, loss_name="total_loss")

    network = fe.Network(ops=[
        ModelOp(inputs='image', model=resunet50, outputs=["label_pred", "mask_pred"]),
        SparseCategoricalCrossentropy(inputs=["label", "label_pred"], outputs="cls_loss"),
        BinaryCrossentropy(inputs=["annotation", "mask_pred"], outputs="seg_loss"),
        ModelOp(inputs=("cls_loss", "seg_loss"), model=uncertainty, outputs="total_loss"),
        Loss(inputs="total_loss", outputs="total_loss")
    ])

    # estimator
    traces = [
        Dice(true_key="annotation", pred_key='mask_pred'),
        Accuracy(true_key="label", pred_key="label_pred"),
        ModelSaver(model_name="resunet50", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=traces,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
