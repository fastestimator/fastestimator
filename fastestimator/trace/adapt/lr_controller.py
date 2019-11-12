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
import numpy as np
from tensorflow.python.keras import backend

from fastestimator.schedule import LRSchedule
from fastestimator.trace import Trace


class LRController(Trace):
    """Learning rate controller that makes learning rate follow the custom schedule and optionally reduces learning
    rate whenever evaluation loss meets certain condition.

    Args:
        model_name (str): Model name of target model
        lr_schedule (object, optional): Scheduler that defines how learning rate changes. It should be `LRSchedule`
            object. Defaults to None.
        reduce_on_eval (bool, optional): If true, it will reduce the learning rate when evaluation loss have been not
            improving for several times. Defaults to False.
        reduce_patience (int, optional): Maximum accumulation of times of not being improving. Defaults to 10.
        reduce_factor (float, optional): Reduce factor of learning rate. Defaults to 0.1.
        reduce_mode (str, optional): It should be {"max", "min"}. If "max", the learning rate will reduce if
            monitored number is too low. If "min", the learning rate will reduce if target is too high. Defaults to
            "min".
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.
    """
    def __init__(self,
                 model_name,
                 lr_schedule=None,
                 reduce_on_eval=False,
                 reduce_patience=10,
                 reduce_factor=0.1,
                 reduce_mode="min",
                 min_lr=1e-6):

        if isinstance(reduce_on_eval, str):
            super().__init__(inputs=reduce_on_eval)
        else:
            super().__init__()
        self.model_name = model_name
        self.lr_schedule = lr_schedule
        self.reduce_on_eval = reduce_on_eval
        self.reduce_patience = reduce_patience
        self.reduce_factor = reduce_factor
        self.reduce_mode = reduce_mode
        self.min_lr = min_lr
        self.reduce_lr_ratio = 1.0
        self.base_lr = None
        self.current_lr = None
        self.log_steps = None
        self.model = None
        self.change_lr = False
        self.wait = 0
        if self.lr_schedule:
            assert isinstance(self.lr_schedule, LRSchedule), "lr_schedule must be instance of LRSchedule"
        if self.reduce_mode == "min":
            self.reduce_metric_best = np.Inf
            self.monitor_op = np.less
        elif self.reduce_mode == "max":
            self.reduce_metric_best = -np.Inf
            self.monitor_op = np.greater
        else:
            raise ValueError("reduce_mode must be either 'min' or 'max'")

    def on_begin(self, state):
        self.log_steps = state["log_steps"]
        self.model = self.network.model[self.model_name]
        self.base_lr = backend.get_value(self.model.optimizer.lr)
        self.current_lr = max(self.base_lr * self.reduce_lr_ratio, self.min_lr)
        if self.reduce_on_eval is True:
            self.reduce_on_eval = self.model.loss_name
        if self.lr_schedule:
            self.lr_schedule.total_epochs = state["total_epochs"]
            self.lr_schedule.total_train_steps = state["total_train_steps"]
            self.lr_schedule.initial_lr = self.current_lr

    def on_epoch_begin(self, state):
        if state["mode"] == "train":
            if self.lr_schedule and self.lr_schedule.schedule_mode == "epoch":
                self.base_lr = self.lr_schedule.schedule_fn(state["epoch"], self.base_lr)
                self.change_lr = True

    def on_batch_begin(self, state):
        if state["mode"] == "train":
            if self.lr_schedule and self.lr_schedule.schedule_mode == "step":
                self.base_lr = self.lr_schedule.schedule_fn(state["train_step"], self.base_lr)
                self.change_lr = True
            if self.change_lr:
                self._update_lr()

    def on_batch_end(self, state):
        if state["mode"] == "train" and self.log_steps and state["train_step"] % self.log_steps == 0:
            state[self.model_name + "_lr"] = round(self.current_lr, 6)

    def on_epoch_end(self, state):
        if state["mode"] == "eval" and self.reduce_on_eval:
            current_value = state[self.reduce_on_eval]
            if self.monitor_op(current_value, self.reduce_metric_best):
                self.reduce_metric_best = current_value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.reduce_patience:
                    self.reduce_lr_ratio *= self.reduce_factor
                    self.change_lr = True
                    self.wait = 0
                    print("FastEstimator-LRController: learning rate reduced by factor of {}".format(
                        self.reduce_factor))

    def _update_lr(self):
        self.current_lr = max(self.base_lr * self.reduce_lr_ratio, self.min_lr)
        backend.set_value(self.model.optimizer.lr, self.current_lr)
        self.change_lr = False
