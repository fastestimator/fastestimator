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
from collections import ChainMap, deque
from typing import Union, Dict, Optional, Iterable, Any, Tuple, List

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader

import fastestimator as fe
from fastestimator import Network
from fastestimator.cli.cli_util import draw
from fastestimator.op import TensorOp
from fastestimator.pipeline2 import torch_to_tf, BasePipeline, TorchPipeline, TensorFlowPipeline
from fastestimator.schedule.epoch_scheduler import Scheduler
from fastestimator.summary import Summary
from fastestimator.trace import Logger, ModelSaver, MonitorLoss, Trace, TrainInfo
from fastestimator.util.util import get_num_devices, per_replica_to_global, to_list


class Estimator:
    pipeline: Scheduler[BasePipeline]
    epochs: int
    steps_per_epoch: Optional[int]
    validation_steps: Optional[int]
    traces: List[Trace]
    log_steps: int
    num_devices: int
    epoch: int
    train_step: int

    def __init__(self,
                 pipeline: Union[BasePipeline, Dict[str, Union[DataLoader, tf.data.Dataset]], Scheduler[BasePipeline]],
                 network: Network,
                 epochs: int,
                 steps_per_epoch: Optional[int] = None,
                 validation_steps: Optional[int] = None,
                 traces: Union[Trace, Iterable[Trace]] = None,
                 log_steps: int = 100):

        if isinstance(pipeline, BasePipeline):
            self.pipeline = Scheduler({0: pipeline})
        elif isinstance(pipeline, dict):
            sample = None
            for val in pipeline.values():
                if sample is None:
                    sample = val
                    assert isinstance(val, (DataLoader, tf.data.Dataset)), \
                        "All pipeline values must be of type DataLoader or tf.data.Dataset"
                assert isinstance(val, type(sample)), "All pipelines must be of the same type"
            if isinstance(sample, DataLoader):
                self.pipeline = Scheduler({0: TorchPipeline(dataloaders=pipeline)})
            else:
                self.pipeline = Scheduler({0: TensorFlowPipeline(dataloaders=pipeline)})
        elif isinstance(pipeline, Scheduler):
            # TODO support scheduling of vanilla pytorch data loaders
            for pipe in pipeline.epoch_dict.values():
                assert isinstance(pipe, BasePipeline), "All scheduled values must extend BasePipeline"
            self.pipeline = pipeline
        else:
            raise ValueError("Unsupported pipeline value")

        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.traces = [] if traces is None else to_list(traces)
        assert log_steps is None or log_steps > 0, "log_steps must be positive or None"
        self.log_steps = log_steps
        self.num_devices = get_num_devices()
        self.epoch = 0
        self.train_step = 0
        self.total_train_steps = 0
        self._prepare_traces()
        self._prepare_network()
        self._warmup()

    def _warmup(self):
        modes = set(*[x.get_modes() for x in self.pipeline.epoch_dict.values()])
        for mode in modes:
            sig_epochs_pipeline = []
            num_examples = {}
            for epoch, pipeline in self.pipeline.epoch_dict.items():
                sig_epochs_pipeline.extend(pipeline.get_signature_epochs(mode))
                num_examples[epoch] = min(pipeline.get_num_examples(mode=mode, epoch=epoch),
                                          self.steps_per_epoch) if self.steps_per_epoch else pipeline.get_num_examples(
                                              mode=mode, epoch=epoch)
            sig_epochs_network = self.network.op_schedule[mode].keys
            signature_epochs = sorted(list(set(sig_epochs_pipeline) | set(sig_epochs_network)))
            elapsed_epochs = np.diff(signature_epochs + [self.epochs])
            for idx, epoch in enumerate(signature_epochs):
                pipeline = self.pipeline.get_current_value(epoch=epoch)
                batch_size = pipeline.get_global_batch_size(mode=mode, epoch=epoch)
                if epoch >= self.epochs:
                    break
                if mode == "train":
                    self.total_train_steps += (num_examples[epoch] * elapsed_epochs[idx]) // batch_size
                itr = pipeline.transform(mode=mode, epoch=epoch)
                batch = next(itr)
                batch = torch_to_tf(batch)
                state = {
                    "mode": mode,
                    "batch_size": batch_size,
                    "local_batch_size": batch_size // self.num_devices,
                    "epoch": tf.convert_to_tensor(self.epoch),
                    "num_examples": pipeline.get_num_examples(mode=mode, epoch=epoch),
                    "warmup": True
                }
                ops = self.network.load_epoch(epoch, mode)
                if fe.distribute_strategy:
                    fe.distribute_strategy.experimental_run_v2(self.network.run_step, args=(batch, ops, state))
                else:
                    self.network.run_step(batch, ops, state)

    def _prepare_network(self):
        modes = list(*[x.get_modes() for x in self.pipeline.epoch_dict.values()])
        self.network.num_devices = self.num_devices
        self.network.prepare(mode_list=modes)

    def _prepare_traces(self):
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
            "num_examples"
            "log_steps",
            "persist_summary",
            "total_epochs",
            "total_train_steps",
            "summary",
            "warmup"
        } | self.network.all_output_keys
        for _, pipeline in self.pipeline.epoch_dict.items():
            available_outputs = available_outputs | pipeline.get_all_output_keys()
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

    def fit(self, summary: Optional[str] = None) -> Optional[Summary]:
        draw()
        return self._main_loop(summary_name=summary)

    def _main_loop(self, summary_name: Optional[str] = None) -> Optional[Summary]:
        try:
            self.train_step = 0
            self._run_traces_on_begin({
                "train_step": self.train_step,
                "num_devices": self.num_devices,
                "log_steps": self.log_steps,
                "persist_summary": bool(summary_name),
                "total_epochs": self.epochs,
                "total_train_steps": self.total_train_steps
            })
            for self.epoch in range(self.epochs):
                self._run_epoch(mode="train")
                self._run_epoch(mode="eval")
        except EarlyStop:
            pass  # On early stopping we still want to run the final traces and then return results
        summary = Summary(summary_name)
        self._run_traces_on_end({"train_step": self.train_step, "epoch": self.epoch, "summary": summary})
        return None if not summary_name else summary

    def _run_epoch(self, mode):
        pipeline = self.pipeline.get_current_value(self.epoch)
        batch_size = pipeline.get_global_batch_size(mode=mode, epoch=self.epoch)
        num_examples = pipeline.get_num_examples(mode=mode, epoch=self.epoch)
        ops = self.network.load_epoch(epoch=self.epoch, mode=mode)
        batch_idx = 0
        self._run_traces_on_epoch_begin({
            "mode": mode, "epoch": self.epoch, "train_step": self.train_step, "num_examples": num_examples
        })
        for batch in pipeline.transform(mode=mode, epoch=self.epoch):
            batch = torch_to_tf(batch)
            self._run_traces_on_batch_begin({
                "mode": mode,
                "epoch": self.epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "local_batch_size": batch_size // self.num_devices
            })

            if fe.distribute_strategy:
                prediction, batch = self._forward_step_parallel(
                    batch,
                    ops,
                    {
                        "mode": mode,
                        "batch_size": batch_size,
                        "local_batch_size": batch_size // self.num_devices,
                        "epoch": tf.convert_to_tensor(self.epoch),
                        "num_examples": num_examples,
                        "warmup": False
                    })
            else:
                prediction = self._forward_step(
                    batch,
                    ops,
                    {
                        "mode": mode,
                        "batch_size": batch_size,
                        "local_batch_size": batch_size // self.num_devices,
                        "epoch": tf.convert_to_tensor(self.epoch),
                        "num_examples": num_examples,
                        "warmup": False
                    })

            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": mode,
                "epoch": self.epoch,
                "train_step": self.train_step,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "local_batch_size": batch_size // self.num_devices,
                "batch": batch,
            })

            if mode == "train":
                self.train_step += 1
            batch_idx += 1
            if self.steps_per_epoch and mode == "train" and batch_idx > self.steps_per_epoch:
                break
            if self.validation_steps and mode == "eval" and batch_idx > self.steps_per_epoch:
                break
        self._run_traces_on_epoch_end({"mode": mode, "epoch": self.epoch, "train_step": self.train_step})

    def _run_traces_on_begin(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            trace.on_begin(trace_state)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_begin(trace_state)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_begin(trace_state)
        self._check_early_exit()

    def _run_traces_on_batch_end(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_end(trace_state)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_end(trace_state)
        self._check_early_exit()

    def _run_traces_on_end(self, state: Dict[str, Any]):
        trace_outputs = {}
        trace_state = ChainMap(trace_outputs, state)
        for trace in self.traces:
            trace.on_end(trace_state)

    def _check_early_exit(self):
        if self.network.stop_training:
            raise EarlyStop

    @tf.function
    def _forward_step(self, batch: Dict[str, Any], ops: Iterable[TensorOp], state: Dict[str, Any]) -> Dict[str, Any]:
        prediction = self.network.run_step(batch, ops, state)
        # expand dimension on scalar value for consistency with distributed training
        for key, value in prediction.items():
            if isinstance(value, tf.Tensor) and value.shape.rank == 0:
                prediction[key] = tf.expand_dims(value, axis=0)
        return prediction

    @tf.function
    def _forward_step_parallel(self, batch: Dict[str, Any], ops: Iterable[TensorOp],
                               state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prediction = fe.distribute_strategy.experimental_run_v2(self.network.run_step, args=(
            batch,
            ops,
            state, ))
        prediction = per_replica_to_global(prediction)
        batch = per_replica_to_global(batch)
        return prediction, batch


class EarlyStop(Exception):
    pass
