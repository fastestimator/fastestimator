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
from collections import ChainMap

import numpy as np
import tensorflow as tf

from fastestimator.estimator.trace import Trace, Logger
from fastestimator.util.util import NonContext, get_gpu_info


class Estimator:
    def __init__(self,
                 pipeline,
                 network,
                 epochs,
                 steps_per_epoch=None,
                 validation_steps=None,
                 traces=None,
                 devices=None,
                 log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.traces = traces
        self.devices = devices
        self.log_steps = log_steps
        self.configure_device()
        self.distribution_strategy = None

    def configure_device(self):
        device_map, num_gpu = get_gpu_info()
        if self.devices is None:
            if num_gpu > 1:
                self.distribution_strategy = tf.distribute.MirroredStrategy()
        elif isinstance(self.devices, (list, int)):
            if isinstance(self.devices, int):
                self.devices = list(range(self.devices))
            else:
                assert len(self.devices) > 1, "please provide more than one gpu when using device list"
            assert len(self.devices) <= num_gpu, "requested {} devices but only {} available".format(len(self.devices), num_gpu)
            if len(self.devices) > 1:
                device_list = []
                for device_idx in self.devices:
                    assert device_idx in device_map, "cannot find key {} in the device map {}".format(device_idx, device_map)
                    device_list.append(device_map[device_idx])
                self.distribution_strategy = tf.distribute.MirroredStrategy(devices=device_list)
        else:
            raise ValueError("unrecognized device input: {}".format(self.devices))

    def fit(self, inputs=None):
        """
        Function to perform training on the estimator

        Args:
            inputs: Path to input data

        Returns:
            None
        """
        self.inputs = inputs
        self._prepare_pipeline()
        self._prepare_network()
        self._prepare_estimator()
        self._warmup()
        self.train()

    def _prepare_pipeline(self):
        self.pipeline.prepare(inputs=self.inputs, distribute_strategy=self.distribution_strategy)
        self.do_eval = "eval" in self.pipeline.mode_list

    def _prepare_network(self):
        self.network.prepare(mode_list=self.pipeline.mode_list, distribution_strategy=self.distribution_strategy)

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        elif not isinstance(self.traces, list):
            self.traces = [self.traces]
        for trace in self.traces:
            assert isinstance(trace, Trace)
        self._add_traces()

    def _add_traces(self):
        #get by default, logger watches all losses
        self.traces.append(Logger(log_steps=self.log_steps, watch_key=self.network.all_losses))

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
        train_step = 0
        self._run_traces_on_begin()
        for epoch in range(self.epochs):
            ds_iter = self.pipeline.dataset_schedule["train"].get_current_value(epoch)
            global_batch_size = self.pipeline.get_global_batch_size(epoch)
            if self.steps_per_epoch:
                max_steps = self.steps_per_epoch
            else:
                max_steps = min(self.pipeline.num_examples["train"]) // global_batch_size
            ops, model_list, loss_list = self.network.load_epoch(epoch, "train")
            epoch_log = {}
            self._run_traces_on_epoch_begin({"mode": "train", "epoch": epoch, "train_step": train_step, "epoch_log": epoch_log})
            for batch_idx in range(max_steps):
                batch = next(ds_iter)
                self._run_traces_on_batch_begin({
                    "mode": "train",
                    "epoch": epoch,
                    "train_step": train_step,
                    "batch_idx": batch_idx,
                    "batch_size": global_batch_size,
                    "batch": batch
                })
                prediction = self.forward_step(batch, ops, model_list, loss_list, {"mode": "train"})
                batch = ChainMap(prediction, batch)
                self._run_traces_on_batch_end({
                    "mode": "train",
                    "epoch": epoch,
                    "train_step": train_step,
                    "batch_idx": batch_idx,
                    "batch_size": global_batch_size,
                    "batch": batch,
                })
                train_step += 1
            self._run_traces_on_epoch_end({
                "mode": "train",
                "epoch": epoch,
                "train_step": train_step,
                "epoch_log": epoch_log
            })
            if self.do_eval:
                self.val(epoch, global_batch_size, train_step)
        self._run_traces_on_end()
        print("FastEstimator: training finished!")

    def val(self, epoch, global_batch_size, train_step):
        ops, model_list, loss_list = self.network.load_epoch(epoch, "eval")
        ds_iter = self.pipeline.dataset_schedule["eval"].get_current_value(epoch)
        if self.validation_steps:
            max_steps = self.validation_steps
        else:
            max_steps = min(self.pipeline.num_examples["eval"]) // global_batch_size
        epoch_log = {}
        self._run_traces_on_epoch_begin({"mode": "eval", "epoch": epoch, "train_step": train_step, "epoch_log": epoch_log})
        for batch_idx in range(max_steps):
            batch = next(ds_iter)
            self._run_traces_on_batch_begin({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "batch": batch
            })
            prediction = self.forward_step(batch, ops, model_list, loss_list, {"mode": "eval"})
            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "batch_idx": batch_idx,
                "batch_size": global_batch_size,
                "batch": batch
            })
        self._run_traces_on_epoch_end({
            "mode": "eval", "epoch": epoch, "train_step": train_step, "epoch_log": epoch_log
        })

    def _run_traces_on_begin(self):
        for trace in self.traces:
            trace.on_begin()

    def _run_traces_on_epoch_begin(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_begin(state)

    def _run_traces_on_batch_begin(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_begin(state)

    def _run_traces_on_batch_end(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_batch_end(state)

    def _run_traces_on_epoch_end(self, state):
        for trace in self.traces:
            if trace.mode is None or state['mode'] in trace.mode:
                trace.on_epoch_end(state)

    def _run_traces_on_end(self):
        for trace in self.traces:
            trace.on_end()

    @tf.function
    def forward_step(self, batch, ops, model_list, loss_list, state):
        prediction = {}
        batch = ChainMap(prediction, batch)
        self.network.run_step(batch, ops, model_list, loss_list, state)
        return prediction
