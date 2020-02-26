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
from typing import List, Set, Union

import pandas as pd

from fastestimator.trace import Trace
from fastestimator.util import Data
from fastestimator.util.util import to_list


class CSVLogger(Trace):
    """Log monitored quantity in CSV file
    Args:
        filename: Output filename.
        monitor_names: List of key names to monitor.
        mode: Restrict the trace to run only on given modes. None will always execute.
    """
    def __init__(self,
                 filename: str,
                 monitor_names: Union[List[str], str],
                 mode: Union[str, Set[str]] = ("eval", "test")):
        super().__init__(inputs=to_list(monitor_names), mode=mode)
        self.filename = filename
        self.data = None

    def on_begin(self, data: Data):
        self.data = {key: [] for key in self.inputs}
        self.data["mode"] = []
        self.data["epoch"] = []

    def on_epoch_end(self, data: Data):
        self.data["mode"].append(self.system.mode)
        self.data["epoch"].append(self.system.epoch_idx)
        for key in self.inputs:
            self.data[key].append(data[key])

    def on_end(self, data: Data):
        df = pd.DataFrame(data=self.data)
        if os.path.exists(self.filename):
            df.to_csv(self.filename, mode='a', index=False)
        else:
            df.to_csv(self.filename, index=False)
