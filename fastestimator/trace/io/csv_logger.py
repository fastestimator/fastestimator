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
import os
from collections import defaultdict
from typing import Iterable, List, Optional, Union

import pandas as pd

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class CSVLogger(Trace):
    """Log monitored quantities in a CSV file.

    Args:
        filename: Output filename.
        monitor_names: List of keys to monitor. If None then all metrics will be recorded.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 filename: str,
                 monitor_names: Optional[Union[List[str], str]] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs="*" if monitor_names is None else monitor_names, mode=mode)
        self.filename = filename
        self.df = None

    def on_begin(self, data: Data) -> None:
        self.df = pd.DataFrame(columns=["mode", "step", "epoch"])

    def on_epoch_end(self, data: Data) -> None:
        tmpdic = defaultdict()
        tmpdic["mode"] = self.system.mode
        tmpdic["step"] = self.system.global_step
        tmpdic["epoch"] = self.system.epoch_idx

        if "*" in self.inputs:
            for key, value in data.read_logs().items():
                tmpdic[key] = value
                if key not in self.df.columns:
                    self.df[key] = ''
        else:
            for key in self.inputs:
                tmpdic[key].append(data[key])
                if key not in self.df.columns:
                    self.df[key] = ''
        for col in self.df.columns:
            if col not in tmpdic.keys():
                tmpdic[col] = ''

        self.df = self.df.append(tmpdic, ignore_index=True)

    def on_batch_end(self, data: Data) -> None:
        if self.system.mode == "train" and self.system.log_steps and (self.system.global_step % self.system.log_steps
                                                                      == 0 or self.system.global_step == 1):
            tmpdic = defaultdict()
            tmpdic["mode"] = self.system.mode
            tmpdic["step"] = self.system.global_step
            tmpdic["epoch"] = self.system.epoch_idx

            if "*" in self.inputs:
                for key, value in data.read_logs().items():
                    tmpdic[key] = value
                    if key not in self.df.columns:
                        self.df[key] = ''
            else:
                for key in self.inputs:
                    tmpdic[key].append(data[key])
                    if key not in self.df.columns:
                        self.df[key] = ''
            for col in self.df.columns:
                if col not in tmpdic.keys():
                    tmpdic[col] = ''

            self.df = self.df.append(tmpdic, ignore_index=True)

    def on_end(self, data: Data) -> None:

        if self.system.mode == "test":
            if os.path.exists(self.filename):
                df1 = pd.read_csv(self.filename)
                self.df = pd.concat([df1, self.df], axis=0, ignore_index=True)

        self.df.to_csv(self.filename, index=False)
