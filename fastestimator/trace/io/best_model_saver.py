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
from fastestimator.util import Data


class BestModelSaver(Trace):
    """save the weights of best model based on evaluation metric

    Args:
        model: model instance
        save_dir: folder path to save model
        metric: eval metric name to monitor, if None, the model's loss will be used
        saved_best_mode: can be 'min' or 'max'. Defaults to 'min'.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 save_dir: str,
                 metric: Optional[str] = None,
                 save_best_mode: str = "min"):
        super().__init__(mode="eval", inputs=metric)
        self.model = model
        self.save_dir = save_dir
        self.metric = metric
        self.save_best_mode = save_best_mode
        if self.save_best_mode == "min":
            self.best = np.Inf
            self.monitor_op = np.less
        elif self.save_best_mode == "max":
            self.best = -np.Inf
            self.monitor_op = np.greater
        else:
            raise ValueError("save_best_mode must be either 'min' or 'max'")

    def on_begin(self, data: Data):
        if not self.metric:
            assert hasattr(self.model, "loss_name"), "model has no loss specified in UpdateOp of Network"
            self.metric = self.model.loss_name

    def on_epoch_end(self, data: Data):
        if self.save_dir and self.monitor_op(data[self.metric], self.best):
            self.best = data[self.metric]
            model_name = "epoch_{}_best_{}".format(self.system.epoch_idx, self.metric)
            save_model(self.model, self.save_dir, model_name)
