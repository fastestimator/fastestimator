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
import time
from typing import Iterable, List, Set, Union

import numpy as np

from fastestimator.backend.get_lr import get_lr
from fastestimator.backend.to_number import to_number
from fastestimator.summary.system import System
from fastestimator.util.data import Data
from fastestimator.util.util import parse_modes, to_list, to_set


class Trace:
    """Trace controls the training loop. User can use `Trace` to customize their own operations.
    Args:
        inputs: A set of keys that this trace intends to read from the state dictionary as inputs
        outputs: A set of keys that this trace intends to write into the system buffer
        mode: Restrict the trace to run only on given modes ('train', 'eval'). None will always
                        execute
    """
    system: System
    inputs: List[str]
    outputs: List[str]
    mode: Set[str]

    def __init__(self,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = parse_modes(to_set(mode))

    def on_begin(self, data: Data):
        """Runs once at the beginning of training
        """
    def on_epoch_begin(self, data: Data):
        """Runs at the beginning of each epoch
        """
    def on_batch_begin(self, data: Data):
        """Runs at the beginning of each batch
        """
    def on_batch_end(self, data: Data):
        """Runs at the end of each batch

        Args:
            data: value fetched by the inputs
        """
    def on_epoch_end(self, data: Data):
        """Runs at the end of each epoch
        """
    def on_end(self, data: Data):
        """Runs once at the end training.
        """


class TrainEssential(Trace):
    """Essential training information for logging during training. Please don't add this trace into an estimator
    manually. An estimator will add it automatically.
    """
    def __init__(self):
        super().__init__(mode="train", inputs=None, outputs=["steps/sec", "epoch_time", "total_time"])
        self.elapse_times = []
        self.train_start = None
        self.epoch_start = None
        self.step_start = None

    def on_begin(self, data: Data):
        self.train_start = time.perf_counter()
        for model in self.system.network.models:
            if hasattr(model, "current_optimizer"):
                data.write_with_log(model.model_name + "_lr", get_lr(model))

    def on_epoch_begin(self, data: Data):
        if self.system.log_steps:
            self.epoch_start = time.perf_counter()
            self.step_start = time.perf_counter()

    def on_batch_begin(self, data: Data):
        self.system.update_global_step()

    def on_batch_end(self, data: Data):
        if self.system.log_steps and (self.system.global_step % self.system.log_steps == 0
                                      or self.system.global_step == 1):
            if self.system.global_step > 1:
                self.elapse_times.append(time.perf_counter() - self.step_start)
                data.write_with_log("steps/sec", round(self.system.log_steps / np.sum(self.elapse_times), 2))
            self.elapse_times = []
            self.step_start = time.perf_counter()

    def on_epoch_end(self, data: Data):
        if self.system.log_steps:
            self.elapse_times.append(time.perf_counter() - self.step_start)
            data.write_with_log("epoch_time", "{} sec".format(round(time.perf_counter() - self.epoch_start, 2)))

    def on_end(self, data: Data):
        data.write_with_log("total_time", "{} sec".format(round(time.perf_counter() - self.train_start, 2)))
        for model in self.system.network.models:
            if hasattr(model, "current_optimizer"):
                data.write_with_log(model.model_name + "_lr", get_lr(model))


class EvalEssential(Trace):
    def __init__(self, loss_keys: Set[str]):
        super().__init__(mode="eval", inputs=loss_keys, outputs=self._configure_outputs(loss_keys))
        self.eval_results = None
        self.best_loss = None
        self.since_best = 0

    @staticmethod
    def _configure_outputs(inputs: Set[str]) -> List[str]:
        outputs = [elem for elem in inputs]
        if len(inputs) == 1:
            outputs.append("min_" + next(iter(inputs)))
            outputs.append("since_best")
        return outputs

    def on_epoch_begin(self, data: Data):
        self.eval_results = None

    def on_batch_end(self, data: Data):
        if self.eval_results is None:
            self.eval_results = {k: [data[k]] for k in self.inputs}
        else:
            for key in self.inputs:
                self.eval_results[key].append(data[key])

    def on_epoch_end(self, data: Data):
        for key, value_list in self.eval_results.items():
            data.write_with_log(key, np.mean(np.array(value_list), axis=0))
        if len(self.inputs) == 1:
            loss_name = self.inputs[0]
            current_loss = data[loss_name]
            if self.best_loss is None or current_loss < self.best_loss:
                self.best_loss = current_loss
                self.since_best = 0
            else:
                self.since_best += 1
            data.write_with_log("min_" + loss_name, self.best_loss)
            data.write_with_log("since_best", self.since_best)


class Logger(Trace):
    """Trace that prints log, please don't add this trace into an estimator manually.

    Args:
        extra_log_keys (set): set of keys to print from system buffer
    """
    def __init__(self, extra_log_keys: Set[str]):
        super().__init__(inputs=extra_log_keys | {"*"})

    def on_begin(self, data: Data):
        if not self.system.mode == "test":
            self._print_message("FastEstimator-Start: step: 1; ", data)

    def on_batch_end(self, data: Data):
        if self.system.mode == "train" and self.system.log_steps and (self.system.global_step % self.system.log_steps \
                == 0 or self.system.global_step == 1):
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data)

    def on_epoch_end(self, data: Data):
        if self.system.mode == "train":
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data, True)
        elif self.system.mode == "eval":
            self._print_message("FastEstimator-Eval: step: {}; ".format(self.system.global_step), data, True)
        elif self.system.mode == "test":
            self._print_message("FastEstimator-Test: ", data, True)

    def on_end(self, data: Data):
        if not self.system.mode == "test":
            self._print_message("FastEstimator-Finish: step: {}; ".format(self.system.global_step), data)

    def _print_message(self, header: str, data: Data, log_epoch: bool = False):
        log_message = header
        if log_epoch:
            log_message += "epoch: {}; ".format(self.system.epoch_idx)
            self.system.write_summary('epoch', self.system.epoch_idx)
        for key, val in data.read_logs(to_set(self.inputs)).items():
            val = to_number(val)
            self.system.write_summary(key, val)
            if val.size > 1:
                log_message += "\n{}:\n{};".format(key, np.array2string(val, separator=','))
            else:
                log_message += "{}: {}; ".format(key, str(val))
        print(log_message)
