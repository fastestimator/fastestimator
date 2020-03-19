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
from typing import Optional

import numpy as np

from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class EarlyStopping(Trace):
    """Stop training when a monitored quantity has stopped improving.
    Args:
        monitor: Quantity to be monitored. Defaults to "loss".
        min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an
            absolute change of less than min_delta, will count as no improvement. Defaults to 0.
        patience: Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        compare: One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored
            has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing.
            Defaults to 'min'.
        baseline: Baseline value for the monitored quantity. Training will stop if the model doesn't
            show improvement over the baseline. Defaults to None.
        mode: Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
    """
    def __init__(self,
                 monitor: Optional[str] = "loss",
                 min_delta: Optional[int] = 0,
                 patience: Optional[int] = 0,
                 compare: str = 'min',
                 baseline: Optional[float] = None,
                 mode: Optional[str] = 'eval'):
        super().__init__(inputs=monitor, mode=mode)

        if len(self.inputs) != 1:
            raise ValueError("EarlyStopping supports only one monitor key")
        if compare not in ['min', 'max']:
            raise ValueError("compare_mode can only be `min` or `max`")

        self.monitored_key = monitor
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best = 0
        self.patience = patience
        self.baseline = baseline
        if compare == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def on_begin(self, data: Data):
        self.wait = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, data: Data):
        current = data[self.monitored_key]
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.system.stop_training = True
                print("FastEstimator-EarlyStopping: '{}' triggered an early stop. Its best value was {} at epoch {}\
                      ".format(self.monitored_key, self.best, self.system.epoch_idx - self.wait))
