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
from collections import deque
from typing import Iterable, List, Optional, Set, Union

import numpy as np

from fastestimator.backend.get_lr import get_lr
from fastestimator.summary.system import System
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import parse_modes, to_list, to_number, to_set


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
    # You can put keys in here to have them automatically added to EvalEssential without the user having to manually add
    # them to the Estimator monitor_names. See BestModelSaver for an example.
    fe_monitor_names: Set[str]

    def __init__(self,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = parse_modes(to_set(mode))
        self.fe_monitor_names = set()  # The use-case here is rare enough that we don't want to add this to the init sig

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
        self.system.mode = 'train'  # Set mode to 'train' for better log visualization
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
        if self.system.mode == "train" and self.system.log_steps and (self.system.global_step % self.system.log_steps
                                                                      == 0 or self.system.global_step == 1):
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


def sort_traces(traces: List[Trace], available_outputs: Optional[Set[str]] = None) -> List[Trace]:
    """Sort traces to attempt to resolve any dependency issues.

    This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
    representation in order to get the slightly better asymptotic runtime complexity.

    Args:
        traces: A list of traces (not inside schedulers) to be sorted.
        available_outputs: What output keys are already available for the traces to use. If None are provided, the
            sorting algorithm will assume that any keys not generated by traces are being provided by the system.
            This results in a less rigorous sorting.

    Returns:
        The sorted list of `traces`.

    Raises:
        AssertionError: If Traces have circular dependencies or require input keys which are not available.
    """
    sorted_traces = []
    trace_outputs = {output for trace in traces for output in trace.outputs}
    if available_outputs is None:
        # Assume that anything not generated by a Trace is provided by the system
        available_outputs = {inp for trace in traces for inp in trace.inputs} - trace_outputs
        weak_sort = True
    else:
        available_outputs = to_set(available_outputs)
        weak_sort = False
    end_traces = deque()
    intermediate_traces = deque()
    intermediate_outputs = set()
    trace_deque = deque(traces)
    while trace_deque:
        trace = trace_deque.popleft()
        ins = set(trace.inputs)
        outs = set(trace.outputs)
        if not ins or isinstance(trace, (TrainEssential, EvalEssential)):
            sorted_traces.append(trace)
            available_outputs |= outs
        elif "*" in ins:
            if outs:
                end_traces.appendleft(trace)
            else:
                end_traces.append(trace)
        elif ins <= available_outputs or (weak_sort and (ins - outs - available_outputs).isdisjoint(trace_outputs)):
            sorted_traces.append(trace)
            available_outputs |= outs
        else:
            intermediate_traces.append(trace)
            intermediate_outputs |= outs

    already_seen = set()
    while intermediate_traces:
        trace = intermediate_traces.popleft()
        ins = set(trace.inputs)
        outs = set(trace.outputs)
        already_seen.add(trace)
        if ins <= available_outputs or (weak_sort and (ins - outs - available_outputs).isdisjoint(trace_outputs)):
            sorted_traces.append(trace)
            available_outputs |= outs
            already_seen.clear()
        elif ins <= (available_outputs | intermediate_outputs):
            intermediate_traces.append(trace)
        else:
            raise AssertionError("The {} trace has unsatisfiable inputs: {}".format(
                type(trace).__name__, ", ".join(ins - (available_outputs | intermediate_outputs))))

        if intermediate_traces and len(already_seen) == len(intermediate_traces):
            raise AssertionError("Dependency cycle detected amongst traces: {}".format(", ".join(
                [type(tr).__name__ for tr in already_seen])))
    sorted_traces.extend(list(end_traces))
    return sorted_traces
