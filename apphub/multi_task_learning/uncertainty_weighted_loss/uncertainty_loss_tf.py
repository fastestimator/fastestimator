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

import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data import cub200
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, ReadMat, ShiftScaleRotate
from fastestimator.op.numpyop.univariate import Normalize, ReadImage, Reshape
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy, Dice


class ReduceLoss(TensorOp):
    """TensorOp to average loss for a batch"""
    def forward(self, data, state):
        return reduce_mean(data)


class UncertaintyWeightedLoss(layers.Layer):
    """Creates Uncertainty weighted loss layer https://arxiv.org/abs/1705.07115
    """
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        self.w1 = self.add_weight(shape=(), initializer='zeros', trainable=True)
        self.w2 = self.add_weight(shape=(), initializer='zeros', trainable=True)

    def call(self, loss_lists):
        return tf.exp(-self.w1) * loss_lists[0] + self.w1 + tf.exp(-self.w2) * loss_lists[1] + self.w2


def UncertaintyLossNet():
    """Creates Uncertainty weighted loss model https://arxiv.org/abs/1705.07115
    """
    l1 = layers.Input(shape=())
    l2 = layers.Input(shape=())
    loss = UncertaintyWeightedLoss()([l1, l2])
    model = Model(inputs=[l1, l2], outputs=loss)
    return model


def ResUnet50(input_shape=(512, 512, 3), num_classes=200):
    """Network Architecture"""
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
    # classification subnet
    label = layers.GlobalMaxPool2D()(C5)
    label = layers.Flatten()(label)
    label = layers.Dense(num_classes, activation='softmax')(label)
    # segmentation subnet
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


def get_estimator(batch_size=8, epochs=50, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    # load CUB200 dataset.
    train_data = cub200.load_data()
    eval_data = train_data.split(0.3)
    test_data = eval_data.split(0.5)

    # step 1, pipeline
    pipeline = fe.Pipeline(
        batch_size=batch_size,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        ops=[
            ReadImage(inputs="image", outputs="image", parent_path=train_data.parent_path),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
            ReadMat(file='annotation', keys="seg", parent_path=train_data.parent_path),
            Delete(keys="annotation"),
            LongestMaxSize(max_size=512, image_in="image", image_out="image", mask_in="seg", mask_out="seg"),
            PadIfNeeded(min_height=512,
                        min_width=512,
                        image_in="image",
                        image_out="image",
                        mask_in="seg",
                        mask_out="seg",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0),
            ShiftScaleRotate(image_in="image",
                             mask_in="seg",
                             image_out="image",
                             mask_out="seg",
                             mode="train",
                             shift_limit=0.2,
                             rotate_limit=15.0,
                             scale_limit=0.2,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0,
                             mask_value=0),
            Sometimes(HorizontalFlip(image_in="image", mask_in="seg", image_out="image", mask_out="seg", mode="train")),
            Reshape(shape=(512, 512, 1), inputs="seg", outputs="seg")
        ])

    # step 2, network
    resunet50 = fe.build(model_fn=ResUnet50, model_names="resunet50", optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    uncertainty = fe.build(model_fn=UncertaintyLossNet,
                           model_names="uncertainty",
                           optimizer_fn=lambda: tf.optimizers.Adam(2e-5))

    network = fe.Network(ops=[
        ModelOp(inputs='image', model=resunet50, outputs=["label_pred", "mask_pred"]),
        CrossEntropy(inputs=["label_pred", "label"], outputs="cls_loss", form="sparse", average_loss=False),
        CrossEntropy(inputs=["mask_pred", "seg"], outputs="seg_loss", form="binary", average_loss=False),
        ModelOp(inputs=["cls_loss", "seg_loss"], model=uncertainty, outputs="total_loss"),
        ReduceLoss(inputs="total_loss", outputs="total_loss"),
        UpdateOp(model=resunet50, loss_name="total_loss"),
        UpdateOp(model=uncertainty, loss_name="total_loss")
    ])

    # step 3, estimator
    traces = [
        Accuracy(true_key="label", pred_key="label_pred"),
        Dice(true_key="seg", pred_key='mask_pred'),
        BestModelSaver(model=resunet50, save_dir=save_dir, metric="total_loss", save_best_mode="min"),
        LRScheduler(model=resunet50, lr_fn=lambda step: cosine_decay(step, cycle_length=26400, init_lr=1e-4))
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=traces,
                             epochs=epochs,
                             max_steps_per_epoch=max_steps_per_epoch,
                             log_steps=500)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
