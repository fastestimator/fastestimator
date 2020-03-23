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
import inspect
from typing import Callable, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import get_lr, set_lr
from fastestimator.summary import System
from fastestimator.trace import Trace
from fastestimator.util import Data


class LRScheduler(Trace):
    """Learning rate scheduler trace that change the learning rate while training

    Args:
        model: model instance
        lr_fn: lr scheduling function that takes either 'epoch' or 'step' as input. For example:
                lr_fn= lambda step: step * 0.01 or lr_fn= lambda epoch: epoch * 0.01
    """
    system: System

    def __init__(self, model: Union[tf.keras.Model, torch.nn.Module], lr_fn: Callable):
        self.model = model
        self.lr_fn = lr_fn
        assert hasattr(lr_fn, "__call__"), "lr_fn must be a function"
        arg = list(inspect.signature(lr_fn).parameters.keys())
        assert len(arg) == 1 and arg[0] in {"step", "epoch"}, "the lr_fn input arg must be either 'step' or 'epoch'"
        self.schedule_mode = arg[0]
        super().__init__(mode="train", outputs=self.model.model_name + "_lr")

    def on_epoch_begin(self, data: Data):
        if self.schedule_mode == "epoch":
            new_lr = np.float32(self.lr_fn(self.system.epoch_idx))
            set_lr(self.model, new_lr)

    def on_batch_begin(self, data: Data):
        if self.schedule_mode == "step":
            new_lr = np.float32(self.lr_fn(self.system.global_step))
            set_lr(self.model, new_lr)

    def on_batch_end(self, data: Data):
        if self.system.log_steps and self.system.global_step % self.system.log_steps == 0:
            current_lr = np.float32(get_lr(self.model))
            data.write_with_log(self.outputs[0], current_lr)
