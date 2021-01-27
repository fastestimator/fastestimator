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

import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.univariate import ChannelTranspose, Hadamard, Normalize
from fastestimator.op.tensorop import UnHadamard
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import Hinge
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class EccLeNet(torch.nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=10, code_length=None) -> None:
        super().__init__()
        conv_kernel = 3
        self.pool_kernel = 2
        self.conv1 = nn.Conv2d(input_shape[0], 32, conv_kernel)
        self.conv2 = nn.Conv2d(32, 64, conv_kernel)
        self.conv3 = nn.Conv2d(64, 64, conv_kernel)
        flat_x = ((((input_shape[1] - (conv_kernel - 1)) // self.pool_kernel) -
                   (conv_kernel - 1)) // self.pool_kernel) - (conv_kernel - 1)
        flat_y = ((((input_shape[2] - (conv_kernel - 1)) // self.pool_kernel) -
                   (conv_kernel - 1)) // self.pool_kernel) - (conv_kernel - 1)
        # Create multiple heads
        code_length = code_length or max(16, 1 << (classes - 1).bit_length())
        n_heads = code_length // 4
        self.heads = nn.ModuleList([nn.Linear(flat_x * flat_y * 64, 16) for _ in range(n_heads)])
        self.heads2 = nn.ModuleList(
            [nn.Linear(in_features=16, out_features=code_length // n_heads) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fn.elu(self.conv1(x))
        x = fn.max_pool2d(x, self.pool_kernel)
        x = fn.elu(self.conv2(x))
        x = fn.max_pool2d(x, self.pool_kernel)
        x = fn.elu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = [fn.elu(head(x)) for head in self.heads]
        x = [torch.tanh(head(e)) for head, e in zip(self.heads2, x)]
        x = torch.cat(x, dim=-1)
        return x


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
            ChannelTranspose(inputs="x", outputs="x"),
            Hadamard(inputs="y", outputs="y_code", n_classes=10)
        ],
        num_process=0)

    # step 2
    model = fe.build(model_fn=lambda: EccLeNet(code_length=code_length), optimizer_fn="adam")
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
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["adv_hinge"])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
