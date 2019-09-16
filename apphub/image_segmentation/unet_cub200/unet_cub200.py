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

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.unet import UNet
from fastestimator.dataset import cub200
from fastestimator.estimator.trace import Dice, ModelSaver
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.pipeline.processing import Minmax
from fastestimator.record.preprocess import ImageReader, MatReader, Reshape, Resize
from fastestimator.util.op import NumpyOp


class SelectDictKey(NumpyOp):
    def forward(self, data, state):
        data = data['seg']
        return data


def get_estimator(batch_size=32, epochs=25, model_dir=tempfile.mkdtemp()):
    # load CUB200 dataset.
    csv_path, path = cub200.load_data()
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "FEdata"),
        train_data=csv_path,
        validation_data=0.2,
        ops=[
            ImageReader(inputs='image', parent_path=path),
            Resize(target_size=(128, 128), keep_ratio=True, outputs='image'),
            MatReader(inputs='annotation', parent_path=path),
            SelectDictKey(),
            Resize((128, 128), keep_ratio=True),
            Reshape(shape=(128, 128, 1), outputs="annotation")
        ])
    #data pipeline
    pipeline = fe.Pipeline(batch_size=batch_size, data=writer, ops=Minmax(inputs='image', outputs='image'))

    #Netowrk
    model = FEModel(model_def=UNet, model_name="unet_cub", optimizer=tf.optimizers.Adam())
    network = fe.Network(ops=[
        ModelOp(inputs='image', model=model, outputs='mask_pred'),
        BinaryCrossentropy(y_true='annotation', y_pred='mask_pred')
    ])

    #estimator
    traces = [
        Dice(true_key="annotation", pred_key='mask_pred'),
        ModelSaver(model_name="unet_cub", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(network=network, pipeline=pipeline, traces=traces, epochs=epochs, log_steps=50)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
