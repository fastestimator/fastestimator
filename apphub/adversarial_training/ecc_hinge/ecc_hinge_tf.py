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

from tensorflow.python.keras.layers import Concatenate, Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.univariate import Hadamard, Normalize
from fastestimator.op.tensorop import UnHadamard
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import Hinge
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def ecc_lenet(input_shape=(32, 32, 3), classes=10, code_length=None):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='elu')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='elu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='elu')(pool2)
    flat = Flatten()(conv3)
    # Create multiple heads
    code_length = code_length or max(16, 1 << (classes - 1).bit_length())
    n_heads = code_length // 4
    heads = [Dense(16, activation='elu')(flat) for _ in range(n_heads)]
    heads2 = [Dense(code_length // n_heads, activation='tanh')(head) for head in heads]
    outputs = Concatenate()(heads2)
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
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            Hadamard(inputs="y", outputs="y_code", n_classes=10)
        ])

    # step 2
    model = fe.build(model_fn=lambda: ecc_lenet(code_length=code_length), optimizer_fn='adam')

    network = fe.Network(ops=[
        Watch(inputs="x", mode=('eval', 'test')),
        ModelOp(model=model, inputs="x", outputs="y_pred_code"),
        Hinge(inputs=("y_pred_code", "y_code"), outputs="base_hinge"),
        UpdateOp(model=model, loss_name="base_hinge"),
        UnHadamard(inputs="y_pred_code", outputs="y_pred", n_classes=10, mode=('eval', 'test')),
        # The adversarial attack:
        FGSM(data="x", loss="base_hinge", outputs="x_adverse_hinge", epsilon=epsilon, mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_adverse_hinge", outputs="y_pred_adv_hinge_code", mode=('eval', 'test')),
        Hinge(inputs=("y_pred_adv_hinge_code", "y_code"), outputs="adv_hinge", mode=('eval', 'test')),
        UnHadamard(inputs="y_pred_adv_hinge_code", outputs="y_pred_adv_hinge", n_classes=10, mode=('eval', 'test')),
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv_hinge", output_name="adversarial_accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="base_hinge", save_best_mode="min", load_best_final=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["adv_hinge"])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
