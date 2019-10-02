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

from fastestimator.trace import Trace
from fastestimator.util.util import is_number, to_list


class CSVLogger(Trace):
    """Log monitored quantity in CSV file manner

    Args:
        filename (str): Output filename.
        monitor_names (list of str, optional): List of key names to monitor. The names can be {"mode", "epoch",
            "train_step", or output names that other traces create}. If None, it will record all. Defaults to None.
        separator (str, optional): Seperator for numbers. Defaults to ", ".
        append (bool, optional): If true, it will write csv file in append mode. Otherwise, it will overwrite the
            existed file. Defaults to False.
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
            execute. Defaults to 'eval'.
    """
    def __init__(self, filename, monitor_names=None, separator=", ", append=False, mode="eval"):
        self.keys = monitor_names if monitor_names is None else to_list(monitor_names)
        super().__init__(inputs="*" if self.keys is None else monitor_names, mode=mode)
        self.separator = separator
        self.file = open(filename, 'a' if append else 'w')
        self.file_empty = os.stat(filename).st_size == 0

    def on_epoch_end(self, state):
        if self.keys is None:
            self._infer_keys(state)
        self._make_header()
        vals = [state.get(key, "") for key in self.keys]
        vals = [str(val.numpy()) if hasattr(val, "numpy") else str(val) for val in vals]
        self.file.write("\n" + self.separator.join(vals))

    def on_end(self, state):
        self.file.flush()
        self.file.close()

    def _infer_keys(self, state):
        monitored_keys = []
        for key, val in state.items():
            if isinstance(val, str) or is_number(val):
                monitored_keys.append(key)
            elif hasattr(val, "numpy") and len(val.numpy().shape) == 1:
                monitored_keys.append(key)
        self.keys = sorted(monitored_keys)

    def _make_header(self):
        if self.file_empty:
            self.file.write(self.separator.join(self.keys))
            self.file_empty = False
