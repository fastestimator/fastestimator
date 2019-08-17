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
from collections import ChainMap

import numpy as np
import tensorflow as tf

from fastestimator.estimator.trace import Trace, TrainLogger


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
        self.log_steps = log_steps
        self.traces = traces
        self.num_gpu = 1

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
        self.pipeline._prepare(inputs=self.inputs)
        self.do_eval = "eval" in self.pipeline.mode_list

    def _prepare_network(self):
        self.network._prepare(mode_list=self.pipeline.mode_list)

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        elif not isinstance(self.traces, list):
            self.traces = [self.traces]
        for trace in self.traces:
            assert isinstance(trace, Trace)
        self._add_traces()

    def _add_traces(self):
        self.traces.insert(0, TrainLogger(log_steps=self.log_steps, num_process=self.num_gpu))

    def _warmup(self):
        mode_list = self.pipeline.mode_list
        for mode in mode_list:
            epochs_pipeline = self.pipeline.dataset_schedule[mode].keys
            epochs_network = self.network.op_schedule[mode].keys
            signature_epochs = list(set(epochs_pipeline) | set(epochs_network))
            state = {"mode": mode}
            for epoch in signature_epochs:
                dataset = self.pipeline.dataset_schedule[mode].get_current_value(epoch)
                batch = next(iter(dataset))
                prediction = {}
                batch = ChainMap(prediction, batch)
                self.network.load_epoch(epoch, mode)
                self.network.run_step(batch, state, warm_up=True)

    def train(self):
        self._run_traces_begin({"mode": "train"})
        train_step = 0
        for epoch in range(self.epochs):
            dataset = self.pipeline.dataset_schedule["train"].get_current_value(epoch)
            if self.steps_per_epoch:
                dataset = dataset.take(self.steps_per_epoch)
            batch_size = self.pipeline._get_batch_size(epoch)
            self.network.load_epoch(epoch, "train")
            self._run_traces_on_epoch_begin({"mode": "train", "epoch": epoch, "train_step": train_step})
            for batch in dataset:
                self._run_traces_on_batch_begin({
                    "mode": "train", "epoch": epoch, "train_step": train_step, "batch_size": batch_size
                })
                prediction, loss = self.forward_step(batch, {"mode": "train"})
                batch = ChainMap(prediction, batch)
                self._run_traces_on_batch_end({
                    "mode": "train",
                    "epoch": epoch,
                    "train_step": train_step,
                    "batch_size": batch_size,
                    "batch": batch,
                    "loss": loss
                })
                train_step += 1
            self._run_traces_on_epoch_end({
                "mode": "train",
                "epoch": epoch,
                "train_step": train_step,
                "loss": np.mean(np.array(self.losses), axis=0)
            })
            if self.do_eval:
                self.val(epoch, batch_size, train_step)
        self._run_traces_end({"mode": "train"})
        print("FastEstimator: training finished!")

    def val(self, epoch, batch_size, train_step):
        self._run_traces_begin({"mode": "eval"})
        self.network.load_epoch(epoch, "eval")
        self._run_traces_on_epoch_begin({"mode": "eval", "epoch": epoch, "train_step": train_step})
        dataset = self.pipeline.dataset_schedule["eval"].get_current_value(epoch)
        if self.validation_steps:
            dataset = dataset.take(self.validation_steps)
        for eval_step, batch in enumerate(dataset):
            self._run_traces_on_batch_begin({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "eval_step": eval_step,
                "batch_size": batch_size
            })
            prediction, loss = self.forward_step(batch, {"mode": "eval"})
            batch = ChainMap(prediction, batch)
            self._run_traces_on_batch_end({
                "mode": "eval",
                "epoch": epoch,
                "train_step": train_step,
                "eval_step": eval_step,
                "batch_size": batch_size,
                "batch": batch,
                "loss": loss
            })
        self._run_traces_on_epoch_end({
            "mode": "eval", "epoch": epoch, "train_step": train_step, "loss": np.mean(np.array(self.losses), axis=0)
        })
        self._run_traces_end({"mode": "eval"})

    def _run_traces_begin(self, state):
        for trace in self.traces:
            trace.begin(state)

    def _run_traces_on_epoch_begin(self, state):
        self.losses = []
        for trace in self.traces:
            trace.on_epoch_begin(state)

    def _run_traces_on_batch_begin(self, state):
        for trace in self.traces:
            trace.on_batch_begin(state)

    def _run_traces_on_batch_end(self, state):
        self.losses.append(state["loss"])
        for trace in self.traces:
            trace.on_batch_end(state)

    def _run_traces_on_epoch_end(self, state):
        output_list = []
        for trace in self.traces:
            metric_output = trace.on_epoch_end(state)
            if state["mode"] == "eval" and metric_output is not None:
                trace_name = type(trace).__name__
                output_list.append((trace_name, metric_output))
        if state["mode"] == "eval":
            self._print_eval_message(output_list, state["train_step"])

    def _run_traces_end(self, state):
        for trace in self.traces:
            trace.end(state)

    @staticmethod
    def _format_log_message(message, metric_name, metric_value):
        if not isinstance(metric_value, np.ndarray):
            log_message = "{}{}: {}; ".format(message, metric_name, str(metric_value))
        else:
            log_message = "{}\n{}:\n{};\n".format(message, metric_name, np.array2string(metric_value, separator=','))
        return log_message

    def _print_eval_message(self, output_list, train_step):
        log_message = "FastEstimator-Eval: step: {}; ".format(train_step)
        for metric_name, metric_result in output_list:
            if isinstance(metric_result, dict):
                for key in metric_result.keys():
                    log_message = self._format_log_message(log_message, key, metric_result[key])
            else:
                log_message = self._format_log_message(log_message, metric_name, metric_result)
        print(log_message)

    @tf.function
    def forward_step(self, batch, state):
        prediction = {}
        batch = ChainMap(prediction, batch)
        losses = self.network.run_step(batch, state)
        return prediction, losses
