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
import re
import time
from collections import defaultdict, deque, namedtuple
from typing import Iterable, List, Set, Union

import numpy as np
from natsort import humansorted

from fastestimator.backend._get_lr import get_lr
from fastestimator.summary.summary import ValWithError
from fastestimator.summary.system import System
from fastestimator.util.base_util import check_ds_id, check_io_names, parse_modes, to_list, to_set
from fastestimator.util.data import Data, DSData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


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
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    system: System
    inputs: List[str]
    outputs: List[str]
    mode: Set[str]
    ds_id: Set[str]
    # You can put keys in here to have them automatically added to EvalEssential without the user having to manually add
    # them to the Estimator monitor_names. See BestModelSaver for an example.
    fe_monitor_names: Set[str]

    def __init__(self,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        self.inputs = check_io_names(to_list(inputs))
        self.outputs = check_io_names(to_list(outputs))
        self.mode = parse_modes(to_set(mode))
        self.ds_id = check_ds_id(to_set(ds_id))
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

    def get_outputs(self, ds_ids: Union[None, str, List[str]]) -> List[str]:
        """What outputs will be generated by this trace.

        You can ignore this unless you are designing a new trace that has special interactions between its outputs and
        particular dataset ids.

        Args:
            ds_ids: The ds_ids under which this trace will execute.

        Returns:
            The outputs of this trace.
        """
        return list(self.outputs)


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
            if self.system.ds_id != '':
                data = DSData(self.system.ds_id, data)
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
            data.write_with_log("epoch_time(sec)", "{}".format(round(time.perf_counter() - self.epoch_start, 2)))

    def on_end(self, data: Data) -> None:
        self.system.mode = 'train'  # Set mode to 'train' for better log visualization
        data.write_with_log("total_time(sec)", "{}".format(round(time.perf_counter() - self.train_start, 2)))
        for model in self.system.network.models:
            if hasattr(model, "current_optimizer") and model.current_optimizer:
                data.write_with_log(model.model_name + "_lr", get_lr(model))


@traceable()
class EvalEssential(Trace):
    """A trace to collect important information during evaluation.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

    Args:
        monitor_names: Any keys which should be collected over the course of an eval epoch.
    """

    def __init__(self, monitor_names: Set[str]) -> None:
        super().__init__(mode="eval", inputs=monitor_names, outputs=["steps/sec"])
        self.step_start = time.perf_counter()
        self.eval_results = defaultdict(lambda: defaultdict(list))
        self.eval_steps = defaultdict(lambda: 0)
        self.elapsed_step = 0

    def on_epoch_begin(self, data: Data) -> None:
        self.eval_results = defaultdict(lambda: defaultdict(list))
        self.eval_steps.clear()
        self.elapsed_step = 0
        self.step_start = time.perf_counter()

    def on_batch_end(self, data: Data) -> None:
        for key in self.inputs:
            if key in data:
                self.eval_results[key][self.system.ds_id].append(data[key])

        self.eval_steps[self.system.ds_id] += 1
        if self.eval_steps[self.system.ds_id] in self.system.eval_log_steps[0]:
            if self.eval_steps[self.system.ds_id] > 1:
                elapsed_time = time.perf_counter() - self.step_start
                elapsed_step = self.eval_steps[self.system.ds_id] - self.elapsed_step
                data.write_with_log("steps/sec", round(elapsed_step / elapsed_time, 2))
            self.elapsed_step = self.eval_steps[self.system.ds_id]
            self.step_start = time.perf_counter()

    def on_epoch_end(self, data: Data) -> None:
        for key, ds_vals in self.eval_results.items():
            for ds_id, vals in ds_vals.items():
                if ds_id != '':
                    d = DSData(ds_id, data)
                    d.write_with_log(key, np.mean(np.array(vals), axis=0))
            data.write_with_log(key, np.mean(np.array([e for x in ds_vals.values() for e in x]), axis=0))


@traceable()
class TestEssential(Trace):
    """A trace to collect important information during evaluation.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

    Args:
        monitor_names: Any keys which should be collected over the course of an test epoch.
    """

    def __init__(self, monitor_names: Set[str]) -> None:
        super().__init__(mode="test", inputs=monitor_names)
        self.test_results = defaultdict(lambda: defaultdict(list))

    def on_epoch_begin(self, data: Data) -> None:
        self.test_results = defaultdict(lambda: defaultdict(list))

    def on_batch_end(self, data: Data) -> None:
        for key in self.inputs:
            if key in data:
                self.test_results[key][self.system.ds_id].append(data[key])

    def on_epoch_end(self, data: Data) -> None:
        for key, ds_vals in self.test_results.items():
            for ds_id, vals in ds_vals.items():
                if ds_id != '':
                    d = DSData(ds_id, data)
                    d.write_with_log(key, np.mean(np.array(vals), axis=0))
            data.write_with_log(key, np.mean(np.array([e for x in ds_vals.values() for e in x]), axis=0))


@traceable()
class Logger(Trace):
    """A Trace that prints log messages.

    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.
    """
    def __init__(self) -> None:
        super().__init__(inputs="*")
        self.eval_steps = defaultdict(lambda: 0)

    def on_begin(self, data: Data) -> None:
        if not self.system.mode == "test":
            start_step = 1 if not self.system.global_step else self.system.global_step
            self._print_message("FastEstimator-Start: step: {}; ".format(start_step), data)

    def on_batch_end(self, data: Data) -> None:
        if self.system.mode == "train" and self.system.log_steps and (self.system.global_step % self.system.log_steps
                                                                      == 0 or self.system.global_step == 1):
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data)

        if self.system.mode == "eval":
            self.eval_steps[self.system.ds_id] += 1
            step = self.eval_steps[self.system.ds_id]
            if step in self.system.eval_log_steps[0]:
                ds_str = f" ({self.system.ds_id})" if self.system.ds_id else ''
                self._print_message(f"Eval Progress{ds_str}: {step}/{self.system.eval_log_steps[1]}; ",
                                    data)

    def on_epoch_end(self, data: Data) -> None:
        if self.system.mode == "train":
            self._print_message("FastEstimator-Train: step: {}; ".format(self.system.global_step), data, True)
        elif self.system.mode == 'eval':
            self.eval_steps.clear()
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
        deferred = []
        for key, val in humansorted(data.read_logs().items(), key=lambda x: x[0]):
            if isinstance(val, ValWithError):
                log_message += "{}: {}; ".format(key, str(val))
            else:
                val = to_number(val)
                if val.size > 1:
                    deferred.append("\n{}:\n{};".format(key, np.array2string(val, separator=',')))
                else:
                    log_message += "{}: {}; ".format(key, str(val))
            self.system.write_summary(key, val)
        log_message = log_message.strip()
        for elem in deferred:
            log_message += elem
        print(log_message)


def sort_traces(traces: List[Trace], ds_ids: List[str], available_outputs: Union[None, str,
                                                                                 Set[str]] = None) -> List[Trace]:
    """Sort traces to attempt to resolve any dependency issues.

    This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
    representation in order to get the slightly better asymptotic runtime complexity.

    Args:
        traces: A list of traces (not inside schedulers) to be sorted.
        ds_ids: The ds_ids currently available to the traces.
        available_outputs: What output keys are already available for the traces to use. If None are provided, the
            sorting algorithm will assume that any keys not generated by traces are being provided by the system.
            This results in a less rigorous sorting.

    Returns:
        The sorted list of `traces`.

    Raises:
        AssertionError: If Traces have circular dependencies or require input keys which are not available.
    """
    sorted_traces = []
    trace_outputs = {output for trace in traces for output in trace.get_outputs(ds_ids=ds_ids)}
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
        outs = set(trace.get_outputs(ds_ids=ds_ids))
        if not ins or isinstance(trace, (TrainEssential, EvalEssential, TestEssential)):
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
        outs = set(trace.get_outputs(ds_ids=ds_ids))
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


class PerDSTrace(Trace):

    def on_ds_begin(self, data: Data) -> None:
        """Runs at the beginning of each dataset.

        Args:
            data: A dictionary through which traces can communicate with each other. Output here will not be logged.
        """
        pass

    def on_ds_end(self, data: Data) -> None:
        """Runs at the beginning of each dataset.

        Args:
            data: A dictionary through which traces can communicate with each other. Output here will be accumulated
                across all available datasets and then logged during on_epoch_end.
        """
        pass


Freq = namedtuple('Freq', ['is_step', 'freq'])


def parse_freq(freq: Union[None, str, int]) -> Freq:
    """A helper function to convert string based frequency inputs into epochs or steps

    Args:
        freq: One of either None, "step", "epoch", "#s", "#e", or #, where # is an integer.

    Returns:
        A `Freq` object recording whether the trace should run on an epoch basis or a step basis, as well as the
        frequency with which it should run.
    """
    if freq is None:
        return Freq(False, 0)
    if isinstance(freq, int):
        if freq < 1:
            raise ValueError(f"Frequency argument must be a positive integer but got {freq}")
        return Freq(True, freq)
    if isinstance(freq, str):
        if freq in {'step', 's'}:
            return Freq(True, 1)
        if freq in {'epoch', 'e'}:
            return Freq(False, 1)
        parts = re.match(r"^([0-9]+)([se])$", freq)
        if parts is None:
            raise ValueError(f"Frequency argument must be formatted like <int><s|e> but got {freq}")
        freq = int(parts[1])
        if freq < 1:
            raise ValueError(f"Frequency argument must be a positive integer but got {freq}")
        return Freq(parts[2] == 's', freq)
    else:
        raise ValueError(f"Unrecognized type passed as frequency: {type(freq)}")
