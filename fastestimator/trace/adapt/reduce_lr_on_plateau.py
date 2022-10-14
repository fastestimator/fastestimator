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
from operator import lt, gt
from typing import Optional, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._get_lr import get_lr
from fastestimator.backend._set_lr import set_lr
from fastestimator.summary.system import System
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class ReduceLROnPlateau(Trace):
    """Reduce learning rate based on evaluation results.

    Args:
        model: A model instance compiled with fe.build.
        metric: The metric name to be monitored. If None, the model's validation loss will be used as the metric.
        patience: Number of epochs to wait before reducing LR again.
        factor: Reduce factor for the learning rate.
        best_mode: Higher is better ("max") or lower is better ("min").
        min_lr: Minimum learning rate.

    Raises:
        AssertionError: If the loss cannot be inferred from the `model` and a `metric` was not provided.
    """
    system: System

    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 metric: Optional[str] = None,
                 patience: int = 10,
                 factor: float = 0.1,
                 best_mode: str = "min",
                 min_lr: float = 1e-6) -> None:
        if not metric:
            assert hasattr(model, "loss_name"), \
                "ReduceLROnPlateau cannot infer model loss name. Provide a metric or use the model in an UpdateOp."
            assert len(model.loss_name) == 1, "the model has more than one losses, please provide the metric explicitly"
            metric = next(iter(model.loss_name))
        super().__init__(mode="eval", inputs=metric, outputs=model.model_name + "_lr")
        self.fe_monitor_names.add(metric)
        self.model = model
        self.patience = patience
        self.factor = factor
        self.best_mode = best_mode
        self.min_lr = min_lr
        self.wait = 0
        if self.best_mode == "min":
            self.best = np.Inf
            self.monitor_op = lt
        elif self.best_mode == "max":
            self.best = -np.Inf
            self.monitor_op = gt
        else:
            raise ValueError("best_mode must be either 'min' or 'max'")

    def on_epoch_end(self, data: Data) -> None:
        if self.monitor_op(data[self.inputs[0]], self.best):
            self.best = data[self.inputs[0]]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.min_lr, np.float32(self.factor * get_lr(self.model)))
                set_lr(self.model, new_lr)
                self.wait = 0
                data.write_with_log(self.outputs[0], new_lr)
                print("FastEstimator-ReduceLROnPlateau: learning rate reduced to {}".format(new_lr))
