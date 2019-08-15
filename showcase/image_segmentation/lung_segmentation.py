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

from fastestimator.dataset import montgomery
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Dice
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.processing import Minmax, Reshape
from fastestimator.record.preprocess import ImageReader, Resize
from fastestimator.record.record import RecordWriter
from fastestimator.architecture import UNet


def get_estimator():

    train_csv_path, eval_cvs_path, path = montgomery.load_and_set_data()
    writer = RecordWriter(
        train_data=train_csv_path, validation_data=eval_cvs_path, ops=[
            ImageReader(grey_scale=True, inputs="imgpath", parent_path=path, outputs="image"),
            ImageReader(grey_scale=True, inputs="mask", parent_path=path, outputs="mask"),
            Resize(inputs="image", target_size=(512, 512), outputs="image"),
            Resize(inputs="mask", target_size=(512, 512), outputs="mask"),
        ])

    pipeline = Pipeline(
        batch_size=8, data=writer, ops=[
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask"),
            Reshape(shape=(512, 512, 1), inputs="image", outputs="image"),
            Reshape(shape=(512, 512, 1), inputs="mask", outputs="mask")
        ])

    model = build(keras_model=UNet("imgpath", "mask",
                                   input_size=(512, 512, 1)), loss=BinaryCrossentropy(inputs=("mask", "pred_segment")),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    network = Network(ops=[ModelOp(inputs="image", model=model, outputs="pred_segment")])

    estimator = Estimator(network=network, pipeline=pipeline, epochs=25, log_steps=20,
                          traces=[Dice("mask", "pred_segment")])

    return estimator


# command to run this script: fastestimator train lung_segmentation.py --inputs /tmp/FE_MONTGOMERY/FE
