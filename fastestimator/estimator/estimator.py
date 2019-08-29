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
import types
from collections import ChainMap, deque

import tensorflow as tf

from fastestimator.estimator.trace import Logger, MonitorLoss, Trace
from fastestimator.util.util import get_num_devices
from fastestimator.util.cli_util import draw


class Estimator:
    def __init__(self,
                 pipeline,
                 network,
                 epochs,
                 steps_per_epoch=None,
                 validation_steps=None,
                 traces=None,
                 log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.traces = traces
        self.log_steps = log_steps
        self.num_devices = get_num_devices()
        if self.num_devices > 1:
            self.distribute_strategy = tf.distribute.MirroredStrategy()
        else:
            self.distribute_strategy = None
        self.train_start = 0
        self.train_step = 0

    def fit(self):
        """
        Function to perform training on the estimator
        """
        draw()
        self._prepare_pipeline()
        self._prepare_network()
        self._prepare_estimator()
        self._warmup()
        self.train()

    def _prepare_pipeline(self):
        self.pipeline.prepare(distribute_strategy=self.distribute_strategy)
        self.do_eval = "eval" in self.pipeline.mode_list

    def _prepare_network(self):
        self.network.prepare(mode_list=self.pipeline.mode_list, distribute_strategy=self.distribute_strategy)

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        elif not isinstance(self.traces, list):
            self.traces = [self.traces]
        self._add_traces()
        for trace in self.traces:
            assert isinstance(trace, Trace)
            trace.network = self.network
        self._sort_traces()

    def _add_traces(self):
        self.traces.insert(0, MonitorLoss())
        if not any(map(lambda x: isinstance(x, Logger), self.traces)):
            self.traces.append(Logger(log_steps=self.log_steps))

    def _sort_traces(self):
        # This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
        # representation in order to get the slightly better asymptotic runtime complexity
        sorted_traces = []
        available_outputs = {
            "num_devices", "mode", "epoch", "train_step", "batch_idx", "batch_size", "batch", "elapsed_time"
        } | set(self.network.all_losses)
        end_traces = deque()

        intermediate_traces = deque()
        intermediate_outputs = set()

        trace_deque = deque(self.traces)
        while len(trace_deque) > 0:
            trace = trace_deque.popleft()
            if not trace.inputs:
                sorted_traces.append(trace)
                available_outputs |= trace.outputs
            elif "*" in trace.inputs:
                if trace.outputs:
                    end_traces.appendleft(trace)
                else:
                    end_traces.append(trace)
            elif trace.inputs <= available_outputs:
                sorted_traces.append(trace)
                available_outputs |= trace.outputs
            else:
                intermediate_traces.append(trace)
                intermediate_outputs |= trace.outputs

        already_seen = set()
        while len(intermediate_traces) > 0:
            trace = intermediate_traces.popleft()
            already_seen.add(trace)
            if trace.inputs <= available_outputs:
                sorted_traces.append(trace)
                available_outputs |= trace.outputs
                already_seen.clear()
            elif trace.inputs <= (available_outputs | intermediate_outputs):
                intermediate_traces.append(trace)
            else:
                raise AssertionError("Trace {} has unsatisfiable inputs: {}".format(
                    type(trace).__name__, ", ".join(trace.inputs - (available_outputs | intermediate_outputs))))
            if 0 < len(already_seen) == len(intermediate_traces):
                raise AssertionError("Dependency cycle detected amongst traces: {}".format(", ".join(
                    [type(tr).__name__ for tr in already_seen])))
        sorted_traces.extend(list(end_traces))
        self.traces = sorted_traces

    def _warmup(self):
        mode_list = self.pipeline.mode_list
        for mode in mode_list:
            epochs_pipeline = self.pipeline.dataset_schedule[mode].keys
            epochs_network = self.network.op_schedule[mode].keys
            signature_epochs = list(set(epochs_pipeline) | set(epochs_network))
            state = {"mode": mode}
            for epoch in signature_epochs:
                ds_iter = self.pipeline.dataset_schedule[mode].get_current_value(epoch)
                batch = next(ds_iter)
                global_batch_size = self.pipeline.get_global_batch_size(epoch)
                state["batch_size"] = global_batch_size
                ops, model_list, loss_list = self.network.load_epoch(epoch, mode)
                self.network.run_step(batch, ops, model_list, loss_list, state, warm_up=True)

    def train(self):
        self.train_start = time.perf_counter()
        self.train_step = 0
        self._run_traces_on_begin({"num_devices": self.num_devices})
        for epoch in range(self.epochs):
            ds_iter = self.pipeline.dataset_schedule["train"].get_sequential_value(epoch)
            global_batch_size = self.pipeline.get_global_batch_size(epoch)
            if self.steps_per_epoch:
                max_steps = self.steps_per_epoch
            else:
                max_steps = min(self.pipeline.num_examples["train"]) // global_batch_size
            ops, model_list, loss_list = self.network.load_epoch(epoch, "train")
            self._run_traces_on_epoch_begin({"mode": "train", "epoch": epoch, "train_step": self.train_step})
            for batch_idx in range(max_steps):
                batch = next(ds_iter)
                self._run_traces_on_batch_begin({
                    "mode": "train",
                    "epoch": epoch,
                    "train_step": self.train_step,
                    "batch_idx": batch_idx,
                    "batch_size": global_batch_size,
                    "batch": types.MappingProxyType(batch)  # A read-only view of the batch data
                })
                prediction = self.forward_step(batch,
                                               ops,
                                               model_list,
                                               loss_list, {
                                                   "mode": "train", "batch_size": global_batch_size
                                               })
                batch = ChainMap(prediction, batch)
                self._run_traces_on_batch_end({
                    "mode": "train",
                    "epoch": epoch,
                    "train_step": self.train_step,
                    "batch_idx": batch_idx,
                    "batch_size": global_batch_size,
                    "batch": batch,
                })
                self.train_step += 1
            self._run_traces_on_epoch_end({"mode": "train", "epoch": epoch, "train_step": self.train_step})
            if self.do_eval:
                self.val(epoch, global_batch_size, self.train_step)
        self._run_traces_on_end({
            "train_step": self.train_step,
            "num_devices": self.num_devices,
            "elapsed_time": time.perf_counter() - self.train_start
        })

    def val(self, epoch, global_batch_size, train_step):
        ops, model_list, loss_list = self.network.load_epoch(epoch, "eval")
        ds_iter = self.pipeline.dataset_schedule["eval"].get_sequential_value(epoch)
        if self.validation_steps:
            max_steps = self.validation_steps
        else:
            max_steps = min(self.pipeline.num_examples["eval"]) // global_batch_size
        self._run_traces_on_epoch_begin({"mode": "eval", "epoch": epoch, "train_step": train_step})
        for batch_idx in range(max_steps):
            batch = next(ds_iter)
            self._run_traces_on_batch_begin({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "batch": types.MappingProxyType(batch)  # A read-only view of the batch data
            })
            prediction = self.forward_step(batch,
                                           ops,
                                           model_list,
                                           loss_list, {
                                               "mode": "eval", "batch_size": global_batch_size
                                           })
            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "batch": batch
            })
        self._run_traces_on_epoch_end({"mode": "eval", "epoch": epoch, "train_step": train_step})

    def _run_traces_on_begin(self, state):
        for trace in self.traces:
            trace.on_begin(state)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_begin(state)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_begin(state)
        self._check_early_exit()

    def _run_traces_on_batch_end(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_end(trace_state)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_end(trace_state)
        self._check_early_exit()

    def _run_traces_on_end(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            trace.on_end(trace_state)

    def _check_early_exit(self):
        if self.network.stop_training:
            self._run_traces_on_end({
                "train_step": self.train_step,
                "num_devices": self.num_devices,
                "elapsed_time": time.perf_counter() - self.train_start
            })
            exit(0)

    @tf.function
    def forward_step(self, batch, ops, model_list, loss_list, state):
        prediction = {}
        batch = ChainMap(prediction, batch)
        self.network.run_step(batch, ops, model_list, loss_list, state)
        return prediction
