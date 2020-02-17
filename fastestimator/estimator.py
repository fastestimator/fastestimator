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
"""Estimator Class."""
from collections import ChainMap, deque

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.cli.cli_util import draw
from fastestimator.schedule.epoch_scheduler import Scheduler
from fastestimator.summary import Summary
from fastestimator.trace import Logger, ModelSaver, MonitorLoss, Trace, TrainInfo
from fastestimator.util.util import get_num_devices, per_replica_to_global


class Estimator:
    """Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wraps
    up `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other training
    necessary information.

    Args:
        pipeline (obj): Pipeline object that defines the data processing workflow. It should be an instance of
            `fastestimator.pipepline.pipeline.Pipeline`
        network (obj): Network object that defines models and their external connection. It should be an instance of
            `fastestimator.network.network.Network`
        epochs (int): Number of epooch to run.
        steps_per_epoch ([type], optional): Number of steps to run for each training session. If None, this will be the
            training example number divided by batch_size. (round down). Defaults to None.
        validation_steps ([type], optional): Number of steps to run for each evaluation session, If None, this will be
            the evaluation example number divided by batch_size (round down). Defaults to None.
        traces (list, optional): List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps (int, optional): Interval steps of logging. Defaults to 100.
    """
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
        self.summary = False
        self.inputs = None
        self.num_devices = get_num_devices()
        self.train_step = 0
        self.train_epoch = 0
        self.total_train_steps = 0
        self.num_examples = {}
        self.do_eval = False
        self._is_initialized = False
        self.mode_list = ["train"]

    def fit(self, summary=None):
        """Function to perform training on the estimator.

        Args:
            summary (str, optional): Experiment name to return. If None, it won't return anything. Defaults to None.

        Returns:
            Experiment object.
        """
        draw()
        if not self._is_initialized:
            self.summary = summary
            self._prepare_pipeline()
            self._prepare_network()
            self._warmup()
            self._prepare_estimator()
            self._is_initialized = True

        return self._start()

    def _prepare_pipeline(self):
        if not isinstance(self.pipeline, Scheduler):
            self.pipeline = Scheduler({0: self.pipeline})
        do_eval = set()
        for _, pipeline in self.pipeline.epoch_dict.items():
            self._configure_single_pipeline(pipeline)
            do_eval.add("eval" in pipeline.mode_list)
        assert len(do_eval) == 1, "inconsistent validation option between pipelines"
        self.do_eval = do_eval.pop()
        if self.do_eval:
            self.mode_list.append("eval")

    def _configure_single_pipeline(self, pipeline):
        pipeline.global_batch_multiplier = get_num_devices()
        pipeline.eval_shuffle = self.validation_steps is not None
        pipeline.prepare()

    def _prepare_network(self):
        self.network.num_devices = self.num_devices
        self.network.prepare(mode_list=self.mode_list)

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        elif not isinstance(self.traces, list):
            self.traces = [self.traces]
        self._add_traces()
        no_save_warning = True
        for trace in self.traces:
            assert isinstance(trace, Trace)
            trace.network = self.network
            if isinstance(trace, ModelSaver):
                no_save_warning = False
        if no_save_warning:
            print("FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.")
        self._sort_traces()

    def _add_traces(self):
        if self.log_steps:
            self.traces.insert(0, TrainInfo())
            if not any(map(lambda x: isinstance(x, Logger), self.traces)):
                self.traces.append(Logger())
        self.traces.insert(0, MonitorLoss())

    def _sort_traces(self):
        # This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
        # representation in order to get the slightly better asymptotic runtime complexity
        sorted_traces = []
        available_outputs = {
            "num_devices",
            "mode",
            "epoch",
            "train_step",
            "batch_idx",
            "batch_size",
            "batch",
            "elapsed_time",
            "local_batch_size",
            "num_examples",
            "log_steps",
            "persist_summary",
            "total_epochs",
            "total_train_steps",
            "summary",
            "warmup"
        } | self.network.all_output_keys
        for _, pipeline in self.pipeline.epoch_dict.items():
            available_outputs = available_outputs | pipeline.all_output_keys
        end_traces = deque()
        intermediate_traces = deque()
        intermediate_outputs = set()
        trace_deque = deque(self.traces)
        while trace_deque:
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
        while intermediate_traces:
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

            if intermediate_traces and len(already_seen) == len(intermediate_traces):
                raise AssertionError("Dependency cycle detected amongst traces: {}".format(", ".join(
                    [type(tr).__name__ for tr in already_seen])))
        sorted_traces.extend(list(end_traces))
        self.traces = sorted_traces

    def _warmup(self):
        self.total_train_steps = 0
        for mode in self.mode_list:
            num_examples_mode = {}
            epochs_pipeline = []
            for epoch, pipeline in self.pipeline.epoch_dict.items():
                num_examples_mode[epoch] = min(pipeline.num_examples[mode])
                epochs_pipeline.extend(pipeline.dataset_schedule[mode].keys)
            self.num_examples[mode] = Scheduler(num_examples_mode)
            epochs_network = self.network.op_schedule[mode].keys
            signature_epochs = sorted(list(set(epochs_pipeline) | set(epochs_network)))
            if mode == "train":
                elapse_epochs = np.diff(signature_epochs + [self.epochs])
                assert np.all(elapse_epochs > 0), "signature epoch is not sorted correctly"
            state = {"mode": mode}
            for idx, epoch in enumerate(signature_epochs):
                pipeline = self.pipeline.get_current_value(epoch)
                ds_iter = pipeline.dataset_schedule[mode].get_current_value(epoch)
                if mode == "train":
                    global_batch_size = pipeline.get_global_batch_size(epoch)
                    if self.steps_per_epoch:
                        max_steps = self.steps_per_epoch
                    else:
                        max_steps = self.num_examples[mode].get_current_value(epoch) // global_batch_size
                    self.total_train_steps += max_steps * elapse_epochs[idx]
                batch = next(ds_iter)
                state["batch_size"] = pipeline.get_global_batch_size(epoch)
                state["local_batch_size"] = state["batch_size"] // self.num_devices
                state["epoch"] = epoch
                state["num_examples"] = self.num_examples[mode].get_current_value(epoch)
                state["warmup"] = True
                ops = self.network.load_epoch(epoch, mode)
                if fe.distribute_strategy:
                    fe.distribute_strategy.experimental_run_v2(self.network.run_step, args=(batch, ops, state))
                else:
                    self.network.run_step(batch, ops, state)

    def _start(self):
        try:
            self.train_step = 0
            self._run_traces_on_begin({
                "train_step": self.train_step,
                "num_devices": self.num_devices,
                "log_steps": self.log_steps,
                "persist_summary": bool(self.summary),
                "total_epochs": self.epochs,
                "total_train_steps": self.total_train_steps
            })
            for self.train_epoch in range(self.epochs):
                self._run_epoch("train")
                if self.do_eval:
                    self._run_epoch("eval")
        except EarlyStop:
            pass  # On early stopping we still want to run the final traces and then return results
        summary = Summary(self.summary)
        self._run_traces_on_end({"train_step": self.train_step, "epoch": self.train_epoch, "summary": summary})
        return None if not self.summary else summary

    def _run_epoch(self, mode):
        pipeline = self.pipeline.get_current_value(self.train_epoch)
        ds_iter = pipeline.dataset_schedule[mode].get_current_value(self.train_epoch)
        global_batch_size = pipeline.get_global_batch_size(self.train_epoch)
        num_examples = self.num_examples[mode].get_current_value(self.train_epoch)
        if self.steps_per_epoch and mode == "train":
            max_steps = self.steps_per_epoch
        elif self.validation_steps and mode == "eval":
            max_steps = self.validation_steps
        elif num_examples > 0:
            max_steps = num_examples // global_batch_size
        else:
            raise ValueError("must specify steps_per_epoch or validations_steps when using generator")
        ops = self.network.load_epoch(self.train_epoch, mode)
        self._run_traces_on_epoch_begin({
            "mode": mode, "epoch": self.train_epoch, "train_step": self.train_step, "num_examples": num_examples
        })
        for batch_idx in range(max_steps):
            batch = next(ds_iter)
            self._run_traces_on_batch_begin({
                "mode": mode,
                "epoch": self.train_epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "local_batch_size": global_batch_size // self.num_devices
            })
            if fe.distribute_strategy:
                prediction, batch = self._forward_step_parallel(
                    batch,
                    ops,
                    {
                        "mode": mode,
                        "batch_size": global_batch_size,
                        "local_batch_size": global_batch_size // self.num_devices,
                        "epoch": tf.convert_to_tensor(self.train_epoch),
                        "num_examples": num_examples,
                        "warmup": False
                    })
            else:
                prediction = self._forward_step(
                    batch,
                    ops,
                    {
                        "mode": mode,
                        "batch_size": global_batch_size,
                        "local_batch_size": global_batch_size // self.num_devices,
                        "epoch": tf.convert_to_tensor(self.train_epoch),
                        "num_examples": num_examples,
                        "warmup": False
                    })
            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": mode,
                "epoch": self.train_epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "local_batch_size": global_batch_size // self.num_devices,
                "batch": batch,
            })
            if mode == "train":
                self.train_step += 1
        self._run_traces_on_epoch_end({"mode": mode, "epoch": self.train_epoch, "train_step": self.train_step})

    def _run_traces_on_begin(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            trace.on_begin(trace_state)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_begin(trace_state)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self, state):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_begin(trace_state)
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
            raise EarlyStop

    @tf.function
    def _forward_step(self, batch, ops, state):
        prediction = self.network.run_step(batch, ops, state)
        # expand dimension on scalar value for consistency with distributed training
        for key, value in prediction.items():
            if isinstance(value, tf.Tensor) and value.shape.rank == 0:
                prediction[key] = tf.expand_dims(value, axis=0)
        return prediction

    @tf.function
    def _forward_step_parallel(self, batch, ops, state):
        prediction = fe.distribute_strategy.experimental_run_v2(self.network.run_step, args=(
            batch,
            ops,
            state, ))
        prediction = per_replica_to_global(prediction)
        batch = per_replica_to_global(batch)
        return prediction, batch


class EarlyStop(Exception):
    pass
