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
from typing import Set, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.schedule.schedule import get_current_items
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_set


@traceable()
class TerminateOnNaN(Trace):
    """End Training if a NaN value is detected. By default (inputs=None) it will monitor all loss values at the end
    of each batch. If one or more inputs are specified, it will only monitor those values. Inputs may be loss keys
    and/or the keys corresponding to the outputs of other traces (ex. accuracy).

    Args:
        monitor_names: key(s) to monitor for NaN values. If None, all loss values will be monitored. "*" will monitor
            all trace output keys and losses.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self, monitor_names: Union[None, str, Set[str]] = None, mode: Union[None, str,
                                                                                     Set[str]] = None) -> None:
        self.monitored_keys = monitor_names if monitor_names is None else to_set(monitor_names)
        super().__init__(inputs=self.monitored_keys, mode=mode)
        self.all_loss_keys = {}
        self.monitored_loss_keys = {}
        self.monitored_trace_keys = {}
        self.in_list = True

    def on_epoch_begin(self, data: Data) -> None:
        self.all_loss_keys = self.system.network.get_loss_keys()
        if self.monitored_keys is None:
            self.monitored_loss_keys = self.all_loss_keys
        elif "*" in self.monitored_keys:
            self.monitored_loss_keys = self.all_loss_keys
            self.monitored_trace_keys = set()

            for trace in get_current_items(self.system.traces, run_modes=self.system.mode, epoch=self.system.epoch_idx):
                self.monitored_trace_keys.update(trace.outputs)
        else:
            self.monitored_loss_keys = self.monitored_keys & self.all_loss_keys
            self.monitored_trace_keys = self.monitored_keys - self.monitored_loss_keys

    def on_batch_end(self, data: Data) -> None:
        for key in self.monitored_loss_keys:
            if key in data:
                if self._check_nan(data[key]):
                    self.system.stop_training = True
                    print("FastEstimator-TerminateOnNaN: NaN Detected in Loss: {}".format(key))
        for key in self.monitored_trace_keys:
            if key in data:
                if self._check_nan(data[key]):
                    self.system.stop_training = True
                    print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))

    def on_epoch_end(self, data: Data) -> None:
        for key in self.monitored_trace_keys:
            if key in data:
                if self._check_nan(data[key]):
                    self.system.stop_training = True
                    print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))

    @staticmethod
    def _check_nan(val: Union[int, float, np.ndarray, tf.Tensor, torch.Tensor]) -> bool:
        if isinstance(val, tf.Tensor):
            return tf.reduce_any(tf.math.is_nan(val)) or tf.reduce_any(tf.math.is_inf(val))
        elif isinstance(val, torch.Tensor):
            return torch.isnan(val).any() or torch.isinf(val).any()
        else:
            return np.isnan(val).any() or np.isinf(val).any()
