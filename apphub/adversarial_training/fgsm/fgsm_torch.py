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

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def get_estimator(epsilon=0.04,
                  epochs=10,
                  batch_size=32,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = cifair10.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            ChannelTranspose(inputs="x", outputs="x"),
        ])

    # step 2
    model = fe.build(model_fn=lambda: LeNet(input_shape=(3, 32, 32)), optimizer_fn="adam")
    network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon),
        ModelOp(model=model, inputs="x_adverse", outputs="y_pred_adv"),
        CrossEntropy(inputs=("y_pred_adv", "y"), outputs="adv_ce"),
        Average(inputs=("base_ce", "adv_ce"), outputs="avg_ce"),
        UpdateOp(model=model, loss_name="avg_ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="adv_ce", save_best_mode="min"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["base_ce", "adv_ce"])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
