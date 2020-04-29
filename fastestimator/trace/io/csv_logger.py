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
from typing import List, Optional, Set, Union

import pandas as pd

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data


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
                 mode: Union[str, Set[str]] = ("eval", "test")) -> None:
        super().__init__(inputs="*" if monitor_names is None else monitor_names, mode=mode)
        self.filename = filename
        self.data = None

    def on_begin(self, data: Data) -> None:
        self.data = defaultdict(list)

    def on_epoch_end(self, data: Data) -> None:
        self.data["mode"].append(self.system.mode)
        self.data["epoch"].append(self.system.epoch_idx)
        if "*" in self.inputs:
            for key, value in data.read_logs().items():
                self.data[key].append(value)
        else:
            for key in self.inputs:
                self.data[key].append(data[key])

    def on_end(self, data: Data) -> None:
        df = pd.DataFrame(data=self.data)
        if os.path.exists(self.filename):
            df.to_csv(self.filename, mode='a', index=False)
        else:
            df.to_csv(self.filename, index=False)
