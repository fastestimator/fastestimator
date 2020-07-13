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
"""Siamese Network implementation using Omniglot dataset"""
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.backend import feed_forward, to_tensor
from fastestimator.dataset.data import omniglot
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import ShiftScaleRotate
from fastestimator.op.numpyop.univariate import Minmax, ReadImage, Reshape
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.adapt import EarlyStopping, LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import Data


def lr_schedule(epoch):
    """Learning rate schedule"""
    lr = 0.0001 * np.power(0.99, epoch)
    return lr


class L2Regularization(TensorOp):
    """Custom layer level L2 Regularization"""
    def __init__(self, inputs, model, outputs, mode="train"):
        self.model = model
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        loss = data
        l2_loss = 0
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                l2_loss += 0.01 * torch.sum(torch.mul(m.weight, m.weight))
            elif isinstance(m, nn.Linear):
                if name == "fc1":
                    l2_loss += 0.0001 * torch.sum(torch.mul(m.weight, m.weight))
        return loss + l2_loss


class OneShotAccuracy(Trace):
    """Trace for calculating one shot accuracy"""
    def __init__(self, dataset, model, N=20, trials=400, mode=("eval", "test"), output_name="one_shot_accuracy"):

        super().__init__(mode=mode, outputs=output_name)
        self.dataset = dataset
        self.model = model
        self.total = 0
        self.correct = 0
        self.output_name = output_name
        self.N = N
        self.trials = trials

    def on_epoch_begin(self, data: Data):
        self.total = 0
        self.correct = 0

    def on_epoch_end(self, data: Data):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _ in range(self.trials):
            img_path = self.dataset.one_shot_trial(self.N)
            input_img = (np.array([
                np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1).reshape((1, 105, 105)) / 255.
                for i in img_path[0]
            ],
                                  dtype=np.float32),
                         np.array([
                             np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1).reshape((1, 105, 105)) / 255.
                             for i in img_path[1]
                         ],
                                  dtype=np.float32))

            input_img = (to_tensor(input_img[0], "torch").to(device), to_tensor(input_img[1], "torch").to(device))

            prediction_score = feed_forward(self.model, input_img, training=False).cpu().detach().numpy()

            if np.argmax(prediction_score) == 0 and prediction_score.std() > 0.01:
                self.correct += 1

            self.total += 1

        data.write_with_log(self.outputs[0], self.correct / self.total)


class SiameseNetwork(nn.Module):
    """Network Architecture"""
    def __init__(self, classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)

    def branch_forward(self, x):
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = fn.relu(x)

        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = fn.relu(x)

        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = fn.relu(x)

        x = self.conv4(x)
        x = fn.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x1 = self.branch_forward(x[0])
        x2 = self.branch_forward(x[1])
        x = torch.abs(x1 - x2)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def get_estimator(epochs=200,
                  batch_size=128,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir=None):
    # step 1. prepare pipeline
    train_data, eval_data = omniglot.load_data(root_dir=data_dir)
    test_data = eval_data.split(0.5)

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="x_a", outputs="x_a", color_flag='gray'),
            ReadImage(inputs="x_b", outputs="x_b", color_flag='gray'),
            Sometimes(
                ShiftScaleRotate(image_in="x_a",
                                 image_out="x_a",
                                 shift_limit=0.05,
                                 scale_limit=0.2,
                                 rotate_limit=10,
                                 mode="train"),
                prob=0.89),
            Sometimes(
                ShiftScaleRotate(image_in="x_b",
                                 image_out="x_b",
                                 shift_limit=0.05,
                                 scale_limit=0.2,
                                 rotate_limit=10,
                                 mode="train"),
                prob=0.89),
            Minmax(inputs="x_a", outputs="x_a"),
            Minmax(inputs="x_b", outputs="x_b"),
            Reshape(shape=(1, 105, 105), inputs="x_a", outputs="x_a"),
            Reshape(shape=(1, 105, 105), inputs="x_b", outputs="x_b")
        ])

    # step 2. prepare model
    model = fe.build(model_fn=SiameseNetwork, model_name="siamese_net", optimizer_fn="adam")

    network = fe.Network(ops=[
        ModelOp(inputs=["x_a", "x_b"], model=model, outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss", form="binary"),
        L2Regularization(inputs="loss", model=model, outputs="loss", mode="train"),
        UpdateOp(model=model, loss_name="loss")
    ])

    # step 3.prepare estimator
    traces = [
        LRScheduler(model=model, lr_fn=lr_schedule),
        Accuracy(true_key="y", pred_key="y_pred"),
        OneShotAccuracy(dataset=eval_data, model=model, output_name='one_shot_accuracy'),
        BestModelSaver(model=model, save_dir=save_dir, metric="one_shot_accuracy", save_best_mode="max"),
        EarlyStopping(monitor="one_shot_accuracy", patience=20, compare='max', mode="eval")
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
