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

from fastestimator.backend._load_model import load_model
from fastestimator.backend._save_model import save_model
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
        load_best_final: Whether to automatically reload the best model (if available) after training.
        save_architecture: Whether to save the full model architecture in addition to the model weights. This option is
            only available for TensorFlow models at present, and will generate a folder containing several files. The
            model can then be re-instantiated even without access to the original code by calling:
            tf.keras.models.load_model(<path to model folder>).

    Raises:
        AssertionError: If a `metric` is not provided and it cannot be inferred from the `model`.
        ValueError: If `save_best_mode` is an unacceptable string, or `save_architecture` is used with a PyTorch model.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 save_dir: str,
                 metric: Optional[str] = None,
                 save_best_mode: str = "min",
                 load_best_final: bool = False,
                 save_architecture: bool = False) -> None:
        if not metric:
            assert hasattr(model, "loss_name"), \
                "BestModelSaver cannot infer model loss name. Provide a metric or use the model in an UpdateOp."
            assert len(model.loss_name) == 1, "the model has more than one losses, please provide the metric explicitly"
            metric = next(iter(model.loss_name))
        super().__init__(mode="eval",
                         inputs=metric,
                         outputs=["since_best_{}".format(metric), "{}_{}".format(save_best_mode, metric)])
        self.fe_monitor_names.add(metric)
        self.model = model
        self.model_name = "{}_best_{}".format(self.model.model_name, self.metric)
        self.save_dir = save_dir
        self.save_best_mode = save_best_mode
        self.load_best_final = load_best_final
        self.save_architecture = save_architecture
        if save_architecture and isinstance(model, torch.nn.Module):
            raise ValueError("Sorry, architecture saving is not currently enabled for PyTorch")
        self.model_path = None
        self.since_best = 0
        if self.save_best_mode == "min":
            self.best = np.Inf
            self.monitor_op = lt
        elif self.save_best_mode == "max":
            self.best = -np.Inf
            self.monitor_op = gt
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
                self.model_path = save_model(model=self.model,
                                             save_dir=self.save_dir,
                                             model_name=self.model_name,
                                             save_architecture=self.save_architecture)
                print("FastEstimator-BestModelSaver: Saved model to {}".format(self.model_path))
        else:
            self.since_best += 1
        data.write_with_log(self.outputs[0], self.since_best)
        data.write_with_log(self.outputs[1], self.best)

    def on_end(self, data: Data) -> None:
        if self.load_best_final and self.model_path:
            print("FastEstimator-BestModelSaver: Restoring model from {}".format(self.model_path))
            load_model(self.model, self.model_path)
