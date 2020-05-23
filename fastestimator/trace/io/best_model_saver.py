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
from typing import Optional, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.save_model import save_model
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class BestModelSaver(Trace):
    """Save the weights of best model based on a given evaluation metric.

    Args:
        model: A model instance compiled with fe.build.
        save_dir: Folder path into which to save the model.
        metric: Eval metric name to monitor. If None, the model's loss will be used.
        save_best_mode: Can be 'min' or 'max'.

    Raises:
        AssertionError: If a `metric` is not provided and it cannot be inferred from the `model`.
        ValueError: If `save_best_mode` is an unacceptable string.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 save_dir: str,
                 metric: Optional[str] = None,
                 save_best_mode: str = "min") -> None:
        if not metric:
            assert hasattr(model, "loss_name"), "cannot infer model loss name, please put the model to UpdateOp first"
            assert len(model.loss_name) == 1, "the model has more than one losses, please provide the metric explicitly"
            metric = next(iter(model.loss_name))
        super().__init__(mode="eval",
                         inputs=metric,
                         outputs=["since_best_{}".format(metric), "{}_{}".format(save_best_mode, metric)])
        self.model = model
        self.save_dir = save_dir
        self.save_best_mode = save_best_mode
        self.since_best = 0
        if self.save_best_mode == "min":
            self.best = np.Inf
            self.monitor_op = np.less
        elif self.save_best_mode == "max":
            self.best = -np.Inf
            self.monitor_op = np.greater
        else:
            raise ValueError("save_best_mode must be either 'min' or 'max'")

    @property
    def metric(self) -> str:
        return self.inputs[0]

    def on_epoch_end(self, data: Data) -> None:
        if self.monitor_op(data[self.metric], self.best):
            self.best = data[self.metric]
            self.since_best = 0
            if self.save_dir:
                model_name = "{}_best_{}".format(self.model.model_name, self.metric)
                model_path = save_model(self.model, self.save_dir, model_name)
                print("FastEstimator-BestModelSaver: Saved model to {}".format(model_path))
        else:
            self.since_best += 1
        data.write_with_log(self.outputs[0], self.since_best)
        data.write_with_log(self.outputs[1], self.best)
