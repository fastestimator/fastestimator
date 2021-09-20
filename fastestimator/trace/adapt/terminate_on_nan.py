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
from typing import Iterable, Set, Union

from fastestimator.backend.check_nan import check_nan
from fastestimator.schedule.schedule import get_current_items
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class TerminateOnNaN(Trace):
    """End Training if a NaN value is detected.

    By default (monitor_names=None) it will monitor all loss values at the end of each batch. If one or more inputs are
    specified, it will only monitor those values. Inputs may be loss keys and/or the keys corresponding to the outputs
    of other traces (ex. accuracy).

    Args:
        monitor_names: key(s) to monitor for NaN values. If None, all loss values will be monitored. "*" will monitor
            all trace output keys and losses.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(
        self,
        monitor_names: Union[None, str, Iterable[str]] = None,
        mode: Union[None, str, Set[str]] = None,
        ds_id: Union[None, str, Iterable[str]] = None,
    ) -> None:
        super().__init__(inputs=monitor_names, mode=mode, ds_id=ds_id)
        self.monitor_keys = {}
        self.in_list = True

    def on_epoch_begin(self, data: Data) -> None:
        if not self.inputs:
            self.monitor_keys = self.system.network.get_loss_keys()
        elif "*" in self.inputs:
            self.monitor_keys = self.system.network.get_loss_keys()
            for trace in get_current_items(self.system.traces, run_modes=self.system.mode, epoch=self.system.epoch_idx):
                self.monitor_keys.update(trace.outputs)
        else:
            self.monitor_keys = self.inputs

    def on_batch_end(self, data: Data) -> None:
        for key in self.monitor_keys:
            if key in data:
                if check_nan(data[key]):
                    self.system.stop_training = True
                    print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))

    def on_epoch_end(self, data: Data) -> None:
        for key in self.monitor_keys:
            if key in data:
                if check_nan(data[key]):
                    self.system.stop_training = True
                    print("FastEstimator-TerminateOnNaN: NaN Detected in: {}".format(key))
