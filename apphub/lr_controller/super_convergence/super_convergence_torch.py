# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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

import fastestimator as fe
from fastestimator.architecture.pytorch import ResNet9
from fastestimator.backend import get_lr
from fastestimator.dataset.data.cifair10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util.util import Suppressor


def super_schedule(step, lr_max, lr_min, mid, end):
    if step < mid:
        lr = step / mid * (lr_max - lr_min) + lr_min  # linear increase from lr_min to lr_max

    elif mid <= step < mid * 2:
        lr = lr_max - (step - mid) / mid * (lr_max - lr_min)  # linear decrease from lr_max to lr_min

    else:
        lr = max(lr_min - (step - 2 * mid) / (end - 2 * mid) * lr_min, 0)  # linear decrease from lr_min to 0

    return lr


def linear_increase(step, min_lr=0.0, max_lr=0.1, num_steps=1000):
    lr = step / num_steps * (max_lr - min_lr) + min_lr
    return lr


def search_max_lr(pipeline, model, network, epochs):
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"), LRScheduler(model=model, lr_fn=lambda step: linear_increase(step))
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=10,
                             log_steps=10)
    print("Running LR range test for super convergence. It will take a while...")
    with Suppressor():
        summary = estimator.fit("LR_range_test")

    best_step = max(summary.history["eval"]["accuracy"].items(), key=lambda k: k[1])[0]
    max_lr = summary.history["train"]["model_lr"][best_step]
    return max_lr


def get_estimator(epochs=24, batch_size=128, lr_epochs=100, max_train_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x"),
            Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.2)
        ])

    # step 2: prepare network
    model = fe.build(model_fn=ResNet9, optimizer_fn="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # get the max learning rate
    lr_max = search_max_lr(pipeline=pipeline, model=model, network=network, epochs=lr_epochs)
    lr_min = lr_max / 40
    print(f"The maximum LR: {lr_max}, and minimun LR: {lr_min}")
    mid_step = int(epochs * 0.45 * len(train_data) / batch_size)
    end_step = int(epochs * len(train_data) / batch_size)

    # reinitialize the model
    model = fe.build(model_fn=ResNet9, optimizer_fn="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: super_schedule(step, lr_max, lr_min, mid_step, end_step))
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
