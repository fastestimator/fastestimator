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
"""This example showcase FastEstimator usage for pytorch users. In this file, we use data loader as data input.
"""
import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from torch.utils.data import DataLoader, Dataset


class MnistDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = {"x": self.x[idx], "y": self.y[idx]}
        return self.transform_data(data)

    def transform_data(self, data):
        data["x"] = np.expand_dims(data["x"], 0)
        data["x"] = np.float32(data["x"] / 255.0)
        data["y"] = np.int64(data["y"])
        return data


def get_dataloader(x, y, shuffle=True):
    dataset = MnistDataset(x=x, y=y)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=shuffle, num_workers=8)
    return loader


def get_estimator():
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    pipeline = fe.Pipeline(train_data=get_dataloader(x=x_train, y=y_train),
                           eval_data=get_dataloader(x=x_eval[:5000], y=y_eval[:5000], shuffle=False),
                           test_data=get_dataloader(x=x_eval[5000:], y=y_eval[5000:], shuffle=False))
    # step 2
    model = fe.build(model=LeNet(), optimizer="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
