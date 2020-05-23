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
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import parse_modes, to_list, to_set


@traceable()
class Trace:
    """Trace controls the training loop. Users can use the `Trace` base class to customize their own functionality.

    Traces are invoked by the fe.Estimator periodically as it runs. In addition to the current data dictionary, they are
    also given a pointer to the current `System` instance which allows access to more information as well as giving the
    ability to modify or even cancel training. The order of function invocations is as follows:

    ``` plot
            Training:                                       Testing:

        on_begin                                            on_begin
            |                                                   |
        on_epoch_begin (train)  <------<                    on_epoch_begin (test)  <------<
            |                          |                        |                         |
        on_batch_begin (train) <----<  |                    on_batch_begin (test) <----<  |
            |                       |  |                        |                      |  |
        on_batch_end (train) >-----^   |                    on_batch_end (test) >------^  |
            |                          ^                        |                         |
        on_epoch_end (train)           |                    on_epoch_end (test) >---------^
            |                          |                        |
        on_epoch_begin (eval)          |                    on_end
            |                          ^
        on_batch_begin (eval) <----<   |
            |                      |   |
        on_batch_end (eval) >-----^    |
            |                          |
        on_epoch_end (eval) >----------^
            |
        on_end
    ```

    Args:
        inputs: A set of keys that this trace intends to read from the state dictionary as inputs.
        outputs: A set of keys that this trace intends to write into the system buffer.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    system: System
    inputs: List[str]
    outputs: List[str]
    mode: Set[str]

    def __init__(self,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = parse_modes(to_set(mode))

    def on_begin(self, data: Data) -> None:
        """Runs once at the beginning of training or testing.

        Args:
            data: A dictionary through which traces can communicate with each other or write values for logging.
        """
        pass

    def on_epoch_begin(self, data: Data) -> None:
        """Runs at the beginning of each epoch.

        Args:
            data: A dictionary through which traces can communicate with each other or write values for logging.
        """
        pass

    def on_batch_begin(self, data: Data) -> None:
        """Runs at the beginning of each batch.

        Args:
            data: A dictionary through which traces can communicate with each other or write values for logging.
        """
        pass

    def on_batch_end(self, data: Data) -> None:
        """Runs at the end of each batch.

        Args:
            data: The current batch and prediction data, as well as any information written by prior `Traces`.
        """
        pass

    def on_epoch_end(self, data: Data) -> None:
        """Runs at the end of each epoch.

        Args:
            data: A dictionary through which traces can communicate with each other or write values for logging.
        """
        pass

    def on_end(self, data: Data) -> None:
        """Runs once at the end training.

        Args:
            data: A dictionary through which traces can communicate with each other or write values for logging.
        """
        pass


@traceable()
class TrainEssential(Trace):
    """A trace to collect important information during training.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

    Args:
        monitor_names: Which keys from the data dictionary to monitor during training.
    """
    def __init__(self, monitor_names: Set[str]) -> None:
        super().__init__(inputs=monitor_names, mode="train", outputs=["steps/sec", "epoch_time", "total_time"])
        self.elapse_times = []
        self.train_start = None
        self.epoch_start = None
        self.step_start = None

    def on_begin(self, data: Data) -> None:
        self.train_start = time.perf_counter()
        data.write_with_log("num_device", self.system.num_devices)
        data.write_with_log("logging_interval", self.system.log_steps)

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.log_steps:
            self.epoch_start = time.perf_counter()
            self.step_start = time.perf_counter()

    def on_batch_end(self, data: Data) -> None:
        if self.system.log_steps and (self.system.global_step % self.system.log_steps == 0
                                      or self.system.global_step == 1):
            for key in self.inputs:
                if key in data:
                    data.write_with_log(key, data[key])
            if self.system.global_step > 1:
                self.elapse_times.append(time.perf_counter() - self.step_start)
                data.write_with_log("steps/sec", round(self.system.log_steps / np.sum(self.elapse_times), 2))
            self.elapse_times = []
            self.step_start = time.perf_counter()

    def on_epoch_end(self, data: Data) -> None:
        if self.system.log_steps:
            self.elapse_times.append(time.perf_counter() - self.step_start)
            data.write_with_log("epoch_time", "{} sec".format(round(time.perf_counter() - self.epoch_start, 2)))

    def on_end(self, data: Data) -> None:
        data.write_with_log("total_time", "{} sec".format(round(time.perf_counter() - self.train_start, 2)))
        for model in self.system.network.models:
            if hasattr(model, "current_optimizer"):
                data.write_with_log(model.model_name + "_lr", get_lr(model))


@traceable()
class EvalEssential(Trace):
    """A trace to collect important information during evaluation.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

    Args:
        monitor_names: Any keys which should be collected over the course of an eval epoch.
    """
    def __init__(self, monitor_names: Set[str]) -> None:
        super().__init__(mode="eval", inputs=monitor_names)
        self.eval_results = None

    def on_epoch_begin(self, data: Data) -> None:
        self.eval_results = None

    def on_batch_end(self, data: Data) -> None:
        if self.eval_results is None:
            self.eval_results = {key: [data[key]] for key in self.inputs if key in data}
        else:
            for key in self.inputs:
                if key in data:
                    self.eval_results[key].append(data[key])

    def on_epoch_end(self, data: Data) -> None:
        for key, value_list in self.eval_results.items():
            data.write_with_log(key, np.mean(np.array(value_list), axis=0))


@traceable()
class Logger(Trace):
    """A Trace that prints log messages.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.
    """
    def __init__(self) -> None:
        super().__init__(inputs="*")

    def on_begin(self, data: Data) -> None:
        if not self.system.mode == "test":
            start_step = 1 if not self.system.global_step else self.system.global_step
            self._print_message("FastEstimator-Start: step: {}; ".format(start_step), data)

    def on_batch_end(self, data: Data) -> None:
        if self.system.mode == "train" and self.system.log_steps and (
                self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1):
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data)

    def on_epoch_end(self, data: Data) -> None:
        if self.system.mode == "train" and self.system.log_steps:
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data, True)
        elif self.system.mode == "eval":
            self._print_message("FastEstimator-Eval: step: {}; ".format(self.system.global_step), data, True)
        elif self.system.mode == "test":
            self._print_message("FastEstimator-Test: step: {}; ".format(self.system.global_step), data, True)

    def on_end(self, data: Data) -> None:
        if not self.system.mode == "test":
            self._print_message("FastEstimator-Finish: step: {}; ".format(self.system.global_step), data)

    def _print_message(self, header: str, data: Data, log_epoch: bool = False) -> None:
        """Print a log message to the screen, and record the `data` into the `system` summary.

        Args:
            header: The prefix for the log message.
            data: A collection of data to be recorded.
            log_epoch: Whether epoch information should be included in the log message.
        """
        log_message = header
        if log_epoch:
            log_message += "epoch: {}; ".format(self.system.epoch_idx)
            self.system.write_summary('epoch', self.system.epoch_idx)
        for key, val in data.read_logs().items():
            val = to_number(val)
            self.system.write_summary(key, val)
            if val.size > 1:
                log_message += "\n{}:\n{};".format(key, np.array2string(val, separator=','))
            else:
                log_message += "{}: {}; ".format(key, str(val))
        print(log_message)
