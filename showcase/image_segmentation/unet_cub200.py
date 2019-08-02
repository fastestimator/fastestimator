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
"""U-Net bird segmentation example.
"""
import os
import tempfile

import tensorflow as tf

from fastestimator.architecture.unet import unet
from fastestimator.dataset import cub200
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Dice
from fastestimator.pipeline.dynamic.preprocess import AbstractPreprocessing, ImageReader, MatReader, Resize
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax, Reshape

DATA_SAVE_PATH = os.path.join(tempfile.gettempdir(), 'CUB200')

# Download CUB200 dataset.
csv_path, path = cub200.load_data(path=DATA_SAVE_PATH)

class Network:
    """Load U-Net and define train and eval ops.
    """
    def __init__(self):
        self.model = unet("image", "annotation")
        self.optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        self.loss = tf.losses.BinaryCrossentropy()

    def train_op(self, batch):
        """Training loop.

        Args:
            batch (`Tensor`): Batch data for training.

        Returns:
            `Tensor`: Network output and loss value.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(batch["image"])
            loss = self.loss(batch["annotation"], predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions, loss

    def eval_op(self, batch):
        """Evaluation loop.

        Args:
            batch (`Tensor`): Batch data for evaluation.

        Returns:
            `Tensor`: Network output and loss value.
        """
        predictions = self.model(batch["image"], training=False)
        loss = self.loss(batch["annotation"], predictions)
        return predictions, loss


class SelectKey(AbstractPreprocessing):
    """Select specific dict value.
    """
    def transform(self, data, feature=None):
        data = data['seg']
        return data


def get_estimator():
    """Generate FastEstimator estimator.

    Returns:
        Estimator object.
    """
    pipeline = Pipeline(
        batch_size=64,
        feature_name=["image", "annotation"],
        train_data=csv_path,
        validation_data=0.2,
        transform_dataset=[
            [ImageReader(parent_path=path),
             Resize((128, 128), keep_ratio=True)],
            [MatReader(parent_path=path), SelectKey(), Resize((128, 128), keep_ratio=True)]
        ],
        transform_train=[[Reshape((128, 128, 3)), Minmax()], [Reshape((128, 128, 1))]])

    traces = [Dice(y_true_key="annotation")]

    estimator = Estimator(network=Network(),
                          pipeline=pipeline,
                          epochs=400,
                          steps_per_epoch=10,
                          validation_steps=1,
                          log_steps=10,
                          traces=traces)
    return estimator
