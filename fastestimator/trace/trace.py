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
"""Trace contains metrics and other information users want to track."""
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend


class Trace:
    """Trace base class. User can use `Trace` to customize their own operations during training, validation and testing.
    The `Network` instance can be accessible by `self.network`. Trace execution order will attempt to be inferred
    whenever possible based on the provided inputs and outputs variables.

    Args:
        inputs (str, list, set): A set of keys that this trace intends to read from the state dictionary as inputs
        outputs (str, list, set): A set of keys that this trace intends to write into the state dictionary
        mode (string): Restrict the trace to run only on given modes ('train', 'eval', 'test'). None will always
                        execute
    """
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.network = None
        self.mode = mode
        self.inputs = set(filter(None, inputs or {})) if not isinstance(inputs, str) else {inputs}
        self.outputs = set(filter(None, outputs or {})) if not isinstance(outputs, str) else {outputs}

    def on_begin(self, state):
        """Runs once at the beginning of training

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "train_step" (int): current global training step starting from 0
                * "num_devices" (int): number of devices(mainly gpu) that are being used, if cpu only, the number is 1
                * "log_steps" (int): how many training steps between logging intervals
                * "persist_summary" (bool): whether to persist the experiment history/summary
                * "total_epochs" (int): how many epochs the training is scheduled to run for
                * "total_train_steps" (int): how many training steps the training is scheduled to run for
                * any keys written by 'on_begin' of previous traces
        """
    def on_epoch_begin(self, state):
        """Runs at the beginning of each epoch of the mode.

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "mode" (str):  current run time mode, can be "train", "eval" or "test"
                * "epoch" (int): current epoch index starting from 0
                * "train_step" (int): current global training step starting from 0
                * "num_examples" (int): total number of examples available for current mode
                * any keys written by 'on_epoch_begin' of previous traces
        """
    def on_batch_begin(self, state):
        """Runs at the beginning of every batch of the mode.

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "mode" (str): current run time mode, can be "train", "eval" or "test"
                * "epoch" (int): current epoch index starting from 0
                * "train_step" (int): current global training step starting from 0
                * "batch_idx" (int): current local step of the epoch starting from 0
                * "batch_size" (int): current global batch size
                * "local_batch_size" (int): current batch size for single device
                * any keys written by 'on_batch_begin' of previous traces
        """
    def on_batch_end(self, state):
        """Runs at the end of every batch of the mode. Anything written to the top level of the state dictionary will be
        printed in the logs. Things written only to the batch sub-dictionary will not be logged

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "mode" (str):  current run time mode, can be "train", "eval" or "test"
                * "epoch" (int): current epoch index starting from 0
                * "train_step" (int): current global training step starting from 0
                * "batch_idx" (int): current local step of the epoch starting from 0
                * "batch_size" (int): current global batch size
                * "batch" (dict): the batch data after the Network execution
                * "local_batch_size" (int): current batch size for single device
                * <loss_name> defined in model (float): loss of current batch (only available when mode is "train")
                * any keys written by 'on_batch_end' of previous traces
        """
    def on_epoch_end(self, state):
        """Runs at the end of every epoch of the mode. Anything written into the state dictionary will be logged

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "mode" (str):  current run time mode, can be "train", "eval" or "test"
                * "epoch" (int): current epoch index starting from 0
                * "train_step" (int): current global training step starting from 0
                * <loss_name> defined in model (float): average loss of the epoch (only available when mode is "eval")
                * any keys written by 'on_epoch_end' of previous traces
        """
    def on_end(self, state):
        """Runs once at the end training. Anything written into the state dictionary will be logged

        Args:
            state (ChainMap): dictionary of run time that has the following key(s):

                * "train_step" (int): current global training step starting from 0
                * "epoch" (int): current epoch index starting from 0
                * "summary" (Experiment): will be returned from estimator.fit() if a summary input was specified
                * any keys written by 'on_end' of previous traces
        """


class MonitorLoss(Trace):
    """Records loss value. Please don't add this trace into an estimator manually. An estimator will add it
    automatically.

    """
    def __init__(self):
        super().__init__()
        self.epochs_since_best = 0
        self.best_loss = None
        self.epoch_losses = []
        self.eval_results = None

    def on_epoch_begin(self, state):
        self.epoch_losses = self.network.epoch_losses
        if state["mode"] == "eval":
            self.eval_results = None

    def on_batch_end(self, state):
        if state["mode"] == "train":
            for key in self.epoch_losses:
                state[key] = self._reduce_loss(state["batch"][key], state["batch_size"])
        elif state["mode"] == "eval":
            if self.eval_results is None:
                self.eval_results = dict(
                    (key, [self._reduce_loss(state["batch"][key], state["batch_size"])]) for key in self.epoch_losses)
            else:
                for key in self.eval_results.keys():
                    self.eval_results[key].append(self._reduce_loss(state["batch"][key], state["batch_size"]))

    def on_epoch_end(self, state):
        if state["mode"] == "eval":
            for key in self.eval_results.keys():
                state[key] = np.mean(np.array(self.eval_results[key]), axis=0)
            if len(self.eval_results) == 1:
                key = list(self.eval_results.keys())[0]
                if self.best_loss is None or state[key] < self.best_loss:
                    self.best_loss = state[key]
                    self.epochs_since_best = 0
                else:
                    self.epochs_since_best += 1
                state["min_" + key] = self.best_loss
                state["since_best_loss"] = self.epochs_since_best

    @staticmethod
    @tf.function
    def _reduce_loss(element_wise_loss, global_batch_size):
        return tf.reduce_sum(element_wise_loss) / global_batch_size


class TrainInfo(Trace):
    """Essential training information for logging during training. Please don't add this trace into an estimator
    manually. An estimator will add it automatically.

    Args:
        log_steps (int): Interval steps of logging
    """
    def __init__(self):
        super().__init__(mode="train")
        self.log_steps = 0
        self.elapse_times = []
        self.num_example = 0
        self.time_start = None
        self.train_start = None
        self.total_train_steps = None

    def on_begin(self, state):
        self.train_start = time.perf_counter()
        self.total_train_steps = state["total_train_steps"]
        self.log_steps = state['log_steps']
        state["total_train_steps"] = self.total_train_steps
        self._get_lr(state)

    def on_epoch_begin(self, state):
        self.time_start = time.perf_counter()

    def on_batch_end(self, state):
        self.num_example += state["batch_size"]
        if state["train_step"] % self.log_steps == 0:
            if state["train_step"] > 0:
                self.elapse_times.append(time.perf_counter() - self.time_start)
                state["examples/sec"] = round(self.num_example / np.sum(self.elapse_times), 1)
                state["progress"] = "{:.1%}".format(state["train_step"] / self.total_train_steps)
            self.elapse_times = []
            self.num_example = 0
            self.time_start = time.perf_counter()

    def on_epoch_end(self, state):
        self.elapse_times.append(time.perf_counter() - self.time_start)

    def on_end(self, state):
        state['total_time'] = "{} sec".format(round(time.perf_counter() - self.train_start, 2))
        self._get_lr(state)

    def _get_lr(self, state):
        for model_name, model in self.network.model.items():
            if hasattr(model.optimizer, "lr"):
                lr = backend.get_value(model.optimizer.lr)
            else:
                lr = backend.get_value(model.optimizer._optimizer.lr)
            state[model_name + "_lr"] = round(lr, 6)
