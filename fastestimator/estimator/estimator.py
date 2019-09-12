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
from collections import ChainMap, deque

import tensorflow as tf

from fastestimator.estimator.trace import Logger, MonitorLoss, Trace, TrainInfo
from fastestimator.util.cli_util import draw
from fastestimator.util.util import get_num_devices


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
        assert log_steps is None or log_steps > 0, "log_steps must be positive or None"
        self.log_steps = log_steps
        self.inputs = None
        self.num_devices = get_num_devices()
        if self.num_devices > 1:
            self.distribute_strategy = tf.distribute.MirroredStrategy()
        else:
            self.distribute_strategy = None
        self.train_step = 0
        self.train_epoch = 0
        self.do_eval = False

    def fit(self):
        """
        Function to perform training on the estimator
        """
        draw()
        self._prepare_pipeline()
        self._prepare_network()
        self._prepare_estimator()
        self._warmup()
        self._start()

    def _prepare_pipeline(self):
        self.pipeline.global_batch_multiplier = get_num_devices()
        self.pipeline.prepare(distribute_strategy=self.distribute_strategy)
        self.do_eval = "eval" in self.pipeline.mode_list

    def _prepare_network(self):
        self.network.num_devices = self.num_devices
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
        if self.log_steps:
            self.traces.insert(0, TrainInfo(log_steps=self.log_steps))
            self.traces.append(Logger(log_steps=self.log_steps))
        self.traces.insert(0, MonitorLoss())

    def _sort_traces(self):
        # This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
        # representation in order to get the slightly better asymptotic runtime complexity
        sorted_traces = []
        available_outputs = {
            "num_devices", "mode", "epoch", "train_step", "batch_idx", "batch_size", "batch", "elapsed_time"
        } | self.pipeline.all_output_keys | self.network.all_output_keys
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
                state["batch_size"] = self.pipeline.get_global_batch_size(epoch)
                ops, model_list, epoch_losses = self.network.load_epoch(epoch, mode)
                if self.distribute_strategy:
                    self.distribute_strategy.experimental_run_v2(
                        self.network.run_step, args=(
                            batch,
                            ops,
                            model_list,
                            epoch_losses,
                            state,
                            True, ))
                else:
                    self.network.run_step(batch, ops, model_list, epoch_losses, state, warm_up=True)

    def _start(self):
        self.train_step = 0
        self._run_traces_on_begin({"num_devices": self.num_devices})
        for self.train_epoch in range(self.epochs):
            self._run_epoch("train")
            if self.do_eval:
                self._run_epoch("eval")
        self._run_traces_on_end({
            "train_step": self.train_step, "epoch": self.train_epoch, "num_devices": self.num_devices
        })

    def _run_epoch(self, mode):
        ds_iter = self.pipeline.dataset_schedule[mode].get_sequential_value(self.train_epoch)
        global_batch_size = self.pipeline.get_global_batch_size(self.train_epoch)
        if self.steps_per_epoch:
            max_steps = self.steps_per_epoch
        else:
            max_steps = min(self.pipeline.num_examples[mode]) // global_batch_size
        ops, model_list, epoch_losses = self.network.load_epoch(self.train_epoch, mode)
        self._run_traces_on_epoch_begin({"mode": mode, "epoch": self.train_epoch, "train_step": self.train_step})
        for batch_idx in range(max_steps):
            batch = next(ds_iter)
            self._run_traces_on_batch_begin({
                "mode": mode,
                "epoch": self.train_epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size
            })
            if self.distribute_strategy:
                prediction, batch = self._forward_step_parallel(batch, ops,
                                                                model_list,
                                                                epoch_losses, {
                                                                    "mode": mode, "batch_size": global_batch_size
                                                                })
            else:
                prediction = self._forward_step(batch,
                                                ops,
                                                model_list,
                                                epoch_losses, {
                                                    "mode": mode, "batch_size": global_batch_size
                                                })
            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": mode,
                "epoch": self.train_epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "batch": batch,
            })
            if mode == "train":
                self.train_step += 1
        self._run_traces_on_epoch_end({"mode": mode, "epoch": self.train_epoch, "train_step": self.train_step})

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
                "train_epoch": self.train_epoch,
                "num_devices": self.num_devices, })
            exit(0)

    @tf.function
    def _forward_step(self, batch, ops, model_list, epoch_losses, state):
        prediction = self.network.run_step(batch, ops, model_list, epoch_losses, state)
        return prediction

    @tf.function
    def _forward_step_parallel(self, batch, ops, model_list, epoch_losses, state):
        prediction = self.distribute_strategy.experimental_run_v2(self.network.run_step,
                                                                  args=(
                                                                      batch,
                                                                      ops,
                                                                      model_list,
                                                                      epoch_losses,
                                                                      state, ))
        prediction = self._per_replica_to_global(prediction)
        batch = self._per_replica_to_global(batch)
        return prediction, batch

    @staticmethod
    def _per_replica_to_global(data):
        new_data = {}
        for key, value in data.items():
            new_data[key] = tf.concat(value.values, axis=0)
        return new_data
