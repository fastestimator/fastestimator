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
import numpy as np
import tensorflow as tf

from fastestimator.trace import Trace


class TerminateOnNaN(Trace):
    """End Training if a NaN value is detected. By default (inputs=None) it will monitor all loss values at the end
    of each batch. If one or more inputs are specified, it will only monitor those values. Inputs may be loss keys
    and/or the keys corresponding to the outputs of other traces (ex. accuracy) but then the other traces must be
    given before TerminateOnNaN in the trace list.

    Args:
        monitor_names (str, list, optional): What key(s) to monitor for NaN values.
                                            - None (default) will monitor all loss values.
                                            - "*" will monitor all state keys and losses.
                                            Defaults to None.
    """
    def __init__(self, monitor_names=None):
        self.monitored_keys = monitor_names if monitor_names is None else set(monitor_names)
        super().__init__(inputs=self.monitored_keys)
        self.all_loss_keys = {}
        self.monitored_loss_keys = {}
        self.monitored_state_keys = {}

    def on_epoch_begin(self, state):
        self.all_loss_keys = set(self.network.epoch_losses)
        if self.monitored_keys is None:
            self.monitored_loss_keys = self.all_loss_keys
        elif "*" in self.monitored_keys:
            self.monitored_loss_keys = self.all_loss_keys
            self.monitored_state_keys = {"*"}
        else:
            self.monitored_loss_keys = self.monitored_keys & self.all_loss_keys
            self.monitored_state_keys = self.monitored_keys - self.monitored_loss_keys

    def on_batch_end(self, state):
        for key in self.monitored_loss_keys:
            loss = state["batch"][key]
            if tf.reduce_any(tf.math.is_nan(loss)):
                self.network.stop_training = True
                print("FastEstimator-TerminateOnNaN: NaN Detected in Loss: {}".format(key))
        for key in state.keys() if "*" in self.monitored_state_keys else self.monitored_state_keys:
            val = state.get(key, None)
            if self._is_floating(val) and tf.reduce_any(tf.math.is_nan(val)):
                self.network.stop_training = True
                print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))

    def on_epoch_end(self, state):
        for key in state.keys() if "*" in self.monitored_state_keys else self.monitored_state_keys:
            val = state.get(key, None)
            if self._is_floating(val) and tf.reduce_any(tf.math.is_nan(val)):
                self.network.stop_training = True
                print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))

    @staticmethod
    def _is_floating(val):
        return isinstance(val, float) or (isinstance(val, tf.Tensor)
                                          and val.dtype.is_floating) or (isinstance(val, np.ndarray)
                                                                         and np.issubdtype(val.dtype, np.floating))
