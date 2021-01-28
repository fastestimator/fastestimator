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
import math
import tempfile

from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.dataset.data import cifair10
from fastestimator.layers.tensorflow import HadamardCode
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def ecc_lenet(input_shape=(32, 32, 3), classes=10, code_length=None):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    flat = Flatten()(conv3)
    # Create multiple heads
    n_heads = code_length // 4 if code_length else math.ceil(classes / 4)
    heads = [Dense(16, activation='relu')(flat) for _ in range(n_heads)]
    # noinspection PyTypeChecker
    outputs = HadamardCode(classes, code_length=code_length)(heads)
    return Model(inputs=inputs, outputs=outputs)


def get_estimator(epsilon=0.04,
                  epochs=20,
                  batch_size=32,
                  code_length=16,
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
        ops=[Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])

    # step 2
    model = fe.build(model_fn=lambda: ecc_lenet(code_length=code_length), optimizer_fn="adam")

    network = fe.Network(ops=[
        Watch(inputs="x", mode=('eval', 'test')),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        UpdateOp(model=model, loss_name="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon, mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_adverse", outputs="y_pred_adv", mode=('eval', 'test')),
        CrossEntropy(inputs=("y_pred_adv", "y"), outputs="adv_ce", mode=('eval', 'test')),
        Average(inputs=("base_ce", "adv_ce"), outputs="avg_ce", mode='eval')
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="avg_ce", save_best_mode="min", load_best_final=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["adv_ce", "avg_ce"])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
