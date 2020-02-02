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
from typing import Any, Callable, Iterable, List, Union

import numpy as np

from fastestimator.backend.to_number import to_number
from fastestimator.util.util import to_list


class Trace:
    """Trace controls the training loop. User can use `Trace` to customize their own operations.
    Args:
        inputs: A set of keys that this trace intends to read from the state dictionary as inputs
        outputs: A set of keys that this trace intends to write into the state dictionary
        mode: Restrict the trace to run only on given modes ('train', 'eval'). None will always
                        execute
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 log_names=None):
        if isinstance(inputs, Iterable) and not isinstance(inputs, str):
            self.inputs = list(inputs)
        else:
            self.inputs = inputs
        if isinstance(mode, Iterable) and not isinstance(mode, str):
            self.mode = set(mode)
        else:
            self.mode = mode
        self.log_names = log_names
        self.system = None

    def on_begin(self):
        """Runs once at the beginning of training
        """

    def on_epoch_begin(self):
        """Runs at the beginning of each epoch
        """

    def on_batch_begin(self):
        """Runs at the beginning of each batch
        """

    def on_batch_end(self, data: List[Any]):
        """Runs at the end of each batch

        Args:
            data: value fetched by the inputs
        """

    def on_epoch_end(self):
        """Runs at the end of each epoch
        """

    def on_end(self):
        """Runs once at the end training.
        """


class TrainEssential(Trace):
    """Essential training information for logging during training. Please don't add this trace into an estimator
    manually. An estimator will add it automatically.
    """
    def __init__(self, monitor_names):
        self.monitor_names = to_list(monitor_names)
        log_names = to_list(monitor_names.union({"steps/sec", "progress", "total_time", "total_steps"}))
        super().__init__(mode="train", inputs=self.monitor_names, log_names=log_names)
        self.elapse_times = []
        self.time_start = None
        self.train_start = None
        self.system = None

    def on_begin(self):
        self.system.add_buffer("total_steps", self.system.total_steps)
        self.train_start = time.perf_counter()

    def on_epoch_begin(self):
        self.time_start = time.perf_counter()

    def on_batch_end(self, data):
        if self.system.global_step % self.system.log_steps == 0:
            for idx, key in enumerate(self.monitor_names):
                self.system.add_buffer(key, data[idx])
            self.elapse_times.append(time.perf_counter() - self.time_start)
            self.system.add_buffer("steps/sec", round(self.system.log_steps / np.sum(self.elapse_times), 1))
            self.system.add_buffer("progress", "{:.1%}".format(self.system.global_step / self.system.total_steps))
            self.elapse_times = []
            self.time_start = time.perf_counter()

    def on_epoch_end(self):
        self.elapse_times.append(time.perf_counter() - self.time_start)

    def on_end(self):
        self.system.add_buffer("total_time", "{} sec".format(round(time.perf_counter() - self.train_start, 2)))


class EvalEssential(Trace):
    def __init__(self, loss_keys):
        self.loss_keys = to_list(loss_keys)
        self._configure_lognames()
        super().__init__(mode="eval", inputs=self.loss_keys, log_names=self.log_names)
        self.eval_results = None
        self.best_loss = None
        self.since_best = 0

    def _configure_lognames(self):
        self.log_names = [elem for elem in self.loss_keys]
        if len(self.loss_keys) == 1:
            self.log_names.append("min_" + self.loss_keys[0])
            self.log_names.append("since_best")

    def on_epoch_begin(self):
        self.eval_results = None

    def on_batch_end(self, data):
        if self.eval_results is None:
            self.eval_results = dict((key, [data[idx]]) for idx, key in enumerate(self.loss_keys))
        else:
            for idx, key in enumerate(self.loss_keys):
                self.eval_results[key].append(data[idx])

    def on_epoch_end(self):
        for key, value_list in self.eval_results.items():
            self.system.add_buffer(key, np.mean(np.array(value_list), axis=0))
        if len(self.loss_keys) == 1:
            loss_name = self.loss_keys[0]
            current_loss = self.system.read_buffer(loss_name)
            if self.best_loss is None or current_loss < self.best_loss:
                self.best_loss = current_loss
                self.since_best = 0
            else:
                self.since_best += 1
            self.system.add_buffer("min_" + loss_name, self.best_loss)
            self.system.add_buffer("since_best", self.since_best)


class Logger(Trace):
    """Trace that prints log, please don't add this trace into an estimator manually.

    Args:
        log_names (set): set of keys to print from system buffer
    """
    def __init__(self, log_names, loss_names):
        super().__init__()
        self.log_names = log_names
        self.loss_names = loss_names
        self.system = None

    def on_begin(self):
        self._print_message("FastEstimator-Start: step: {}; ".format(self.system.global_step))

    def on_batch_end(self, data):
        if self.system.mode == "train" and self.system.global_step % self.system.log_steps == 0:
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step))

    def on_epoch_end(self):
        if self.system.mode == "eval":
            self._print_message("FastEstimator-Eval: step: {}; ".format(self.system.global_step), True)

    def on_end(self):
        self._print_message("FastEstimator-Finish: step: {}; ".format(self.system.global_step))

    def _print_message(self, header, log_epoch=False):
        log_message = header
        if log_epoch:
            log_message += "epoch: {}; ".format(self.system.epoch_idx)
        for key, val in self.system.buffer.items():
            if key in self.log_names:
                val = to_number(val)
                if key in self.loss_names:
                    val = np.round(val, decimals=7)
                if isinstance(val, np.ndarray):
                    log_message += "\n{}:\n{};".format(key, np.array2string(val, separator=','))
                else:
                    log_message += "{}: {}; ".format(key, str(val))
        print(log_message)
