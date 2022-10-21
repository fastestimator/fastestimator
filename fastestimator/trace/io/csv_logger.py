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
from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from fastestimator.summary import ValWithError
from fastestimator.trace.trace import Trace
from fastestimator.util.base_util import to_list
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@traceable()
class CSVLogger(Trace):
    """Log monitored quantities in a CSV file.

    Args:
        filename: Output filename.
        monitor_names: List of keys to monitor. If None then all metrics will be recorded. If you want to record 'all
            the usual stuff' plus a particular key which isn't normally recorded, you can use a '*' character here.
            For example: monitor_names=['*', 'y_true']
        instance_id_key: A key corresponding to data instance ids. If provided, the CSV logger will record per-instance
            metric information into the csv file in addition to the standard metrics.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 filename: str,
                 monitor_names: Optional[Union[List[str], str]] = None,
                 instance_id_key: Optional[str] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        self.instance_id_key = instance_id_key
        monitor_names = to_list(monitor_names)
        instance_id_key = to_list(instance_id_key)
        inputs = monitor_names if monitor_names else ["*"]
        inputs.extend(instance_id_key)
        super().__init__(inputs=inputs, mode=mode)
        self.filename = filename
        self.df_agg = None  # DataFrame for aggregate metrics
        self.df_ins = None  # DataFrame for instance metrics

    def on_begin(self, data: Data) -> None:
        base_keys = ["instance_id", "mode", "step", "epoch"] if self.instance_id_key else ["mode", "step", "epoch"]
        self.df_agg = pd.DataFrame(columns=base_keys)
        self.df_ins = pd.DataFrame(columns=base_keys)

    def on_epoch_end(self, data: Data) -> None:
        keys = set(self.inputs) if "*" not in self.inputs else set(self.inputs) | data.read_logs().keys()
        keys = keys - {'*', self.instance_id_key}

        tmpdic = {}
        for key in keys:
            tmpdic[key] = self._parse_val(data.read_logs().get(key, ''))
            if key not in self.df_agg.columns:
                self.df_agg[key] = ''
        for col in set(self.df_agg.columns) - {'mode', 'step', 'epoch'} - tmpdic.keys():
            tmpdic[col] = ''

        # Only record an entry if there is at least one piece of actual information present
        if any(tmpdic.values()):
            self.df_agg = pd.concat(objs=[self.df_agg, pd.DataFrame([{"mode": self.system.mode,
                                                                      "step": self.system.global_step,
                                                                      "epoch": self.system.epoch_idx,
                                                                      **tmpdic}])],
                                    ignore_index=True)
        self._save()  # Write on epoch end so that people can see results sooner if debugging

    def on_batch_end(self, data: Data) -> None:
        if self.instance_id_key:
            ins_data = data.read_per_instance_logs()
            keys = set(self.inputs) if "*" not in self.inputs else set(self.inputs) | ins_data.keys()
            keys = list(keys - {'*', self.instance_id_key})
            ids = data[self.instance_id_key]
            batch_size = len(ids)
            vals = [ins_data.get(key, data.get(key, _SKIP())) for key in keys]
            # Ignore vals which are not batched
            vals = [val if (hasattr(val, 'ndim') and val.ndim > 0 and val.shape[0] == batch_size) or
                           (isinstance(val, (list, tuple)) and len(val) == batch_size) else _SKIP() for val in vals]
            # Don't bother recording instance if no data is available
            if any((not isinstance(val, _SKIP) for val in vals)):
                for key in keys:
                    if key not in self.df_ins.columns:
                        self.df_ins[key] = ''
                rows = []
                for sample in zip_longest(ids, *vals, fillvalue=''):
                    row = {"instance_id": self._parse_val(sample[0]),
                           "mode": self.system.mode,
                           "step": self.system.global_step,
                           "epoch": self.system.epoch_idx,
                           **{key: self._parse_val(val) for key, val in zip(keys, sample[1:])}}
                    for col in self.df_ins.columns:
                        if col not in row.keys():
                            row[col] = ''
                    rows.append(row)
                self.df_ins = pd.concat(objs=[self.df_ins, pd.DataFrame(rows)], ignore_index=True)

        if self.system.mode == "train" and self.system.log_steps and (self.system.global_step % self.system.log_steps
                                                                      == 0 or self.system.global_step == 1):

            keys = set(self.inputs) if "*" not in self.inputs else set(self.inputs) | data.read_logs().keys()
            keys = keys - {'*', self.instance_id_key}

            tmpdic = {}
            for key in keys:
                if self.instance_id_key:
                    # If you are using an instance_id key, then don't report per-instance values at the agg level
                    tmpdic[key] = self._parse_val(data.read_logs().get(key, ''))
                else:
                    tmpdic[key] = self._parse_val(data.get(key, ''))
                if key not in self.df_agg.columns:
                    self.df_agg[key] = ''
            for col in set(self.df_agg.columns) - {'mode', 'step', 'epoch'} - tmpdic.keys():
                tmpdic[col] = ''
            # Only record an entry if there's at least 1 piece of actual information
            if any(tmpdic.values()):
                self.df_agg = pd.concat(objs=[self.df_agg, pd.DataFrame([{"mode": self.system.mode,
                                                                          "step": self.system.global_step,
                                                                          "epoch": self.system.epoch_idx,
                                                                          **tmpdic}])],
                                        ignore_index=True)

    def _save(self) -> None:
        """Write the current state to disk.
        """
        stack = [self.df_ins, self.df_agg]
        if self.system.mode == "test":
            if os.path.exists(self.filename):
                df1 = pd.read_csv(self.filename, dtype=str)
                stack.insert(0, df1)
        stack = pd.concat(stack, axis=0, ignore_index=True)
        stack.to_csv(self.filename, index=False)

    @staticmethod
    def _parse_val(val: Any) -> str:
        """Convert values into string representations.

        Args:
            val: A value to be printed.

        Returns:
            A formatted version of `val` appropriate for a csv file.
        """
        if isinstance(val, str):
            return val
        if isinstance(val, ValWithError):
            return str(val).replace(',', ';')
        val = to_number(val)
        if val.size > 1:
            return np.array2string(val, separator=';')
        if val.dtype.kind in {'U', 'S'}:  # Unicode or String
            # remove the b'' from strings stored in tensors
            return str(val, 'utf-8')
        return str(val)


class _SKIP:
    # A class to signify that data should be ignored. Use this rather than None to work better with zip methods
    def __iter__(self):
        return iter(())
