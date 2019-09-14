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
import os
import tempfile

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import UNet
from fastestimator.dataset import montgomery
from fastestimator.estimator.trace import Dice, ModelSaver
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.pipeline.augmentation import Augmentation2D
from fastestimator.pipeline.processing import Minmax
from fastestimator.record.preprocess import ImageReader, Reshape, Resize
from fastestimator.util.op import NumpyOp


class CombineLeftRightMask(NumpyOp):
    def forward(self, data, state):
        mask_left, mask_right = data
        data = mask_left + mask_right
        return data


def get_estimator(batch_size=4, epochs=25, model_dir=tempfile.mkdtemp()):
    csv_path, path = montgomery.load_data()
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "FEdata"),
        train_data=csv_path,
        validation_data=0.2,
        ops=[
            ImageReader(grey_scale=True, inputs="image", parent_path=path, outputs="image"),
            ImageReader(grey_scale=True, inputs="mask_left", parent_path=path, outputs="mask_left"),
            ImageReader(grey_scale=True, inputs="mask_right", parent_path=path, outputs="mask_right"),
            CombineLeftRightMask(inputs=("mask_left", "mask_right")),
            Resize(target_size=(512, 512)),
            Reshape(shape=(512, 512, 1), outputs="mask"),
            Resize(inputs="image", target_size=(512, 512)),
            Reshape(shape=(512, 512, 1), outputs="image"),
        ],
        write_feature=["image", "mask"])

    pipeline = fe.Pipeline(
        batch_size=batch_size,
        data=writer,
        ops=[
            Augmentation2D(inputs=["image", "mask"],
                           outputs=["image", "mask"],
                           mode="train",
                           rotation_range=10,
                           flip_left_right=True),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask")
        ])

    model = FEModel(model_def=lambda: UNet(input_size=(512, 512, 1)),
                    model_name="lungsegmentation",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="pred_segment"),
        BinaryCrossentropy(y_true="mask", y_pred="pred_segment")
    ])

    traces = [
        Dice(true_key="mask", pred_key="pred_segment"),
        ModelSaver(model_name="lungsegmentation", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, log_steps=20, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
