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
"""U-Net bird segmentation example."""
import os
import tempfile

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.unet import UNet
from fastestimator.dataset import cub200
from fastestimator.estimator.trace import Dice
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.pipeline.processing import Minmax, Reshape
from fastestimator.record.preprocess import ImageReader, MatReader, Resize
from fastestimator.util.op import NumpyOp

    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data, state):
        data = data['seg']
        return data


def get_estimator():
    # Download CUB200 dataset.
    data_save_path = os.path.join(tempfile.gettempdir(), 'CUB200')
    csv_path, path = cub200.load_data(path=data_save_path)

    tfr_writer = fe.RecordWriter(
        train_data=os.path.join(path, csv_path),
        validation_data=0.2,
        ops=[
            ImageReader(inputs='image', parent_path=path, outputs='image'),
            Resize(inputs='image', target_size=(128, 128), keep_ratio=True, outputs='image'),
            MatReader(inputs='annotation', parent_path=path),
            SelectDictKey(),
            Resize((128, 128), keep_ratio=True, outputs='annotation')
        ])

    pipeline = fe.Pipeline(batch_size=32, data=tfr_writer, ops=[
    pipeline = fe.Pipeline(
        batch_size=32,
        data=tfr_writer,
        ops=[
            Reshape((128, 128, 3), inputs='image'),
            Minmax(outputs='image'),
            Reshape((128, 128, 1), inputs='annotation', outputs='annotation')
        ])
    model = build(keras_model=UNet('image', 'annotation'),
                  loss=BinaryCrossentropy(y_true='annotation', y_pred='mask_pred'),
                  optimizer=tf.optimizers.Adam(learning_rate=0.0001))

    network = fe.Network(ops=ModelOp(inputs='image', model=model, outputs='mask_pred'))

    traces = [Dice(true_key="annotation", pred_key='mask_pred')]

    estimator = fe.Estimator(network=network, pipeline=pipeline, traces=traces, epochs=400, steps_per_epoch=10)

    return estimator
