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

from fastestimator.trace import Trace


class EarlyStopping(Trace):
    """Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str, optional): Quantity to be monitored.. Defaults to "loss".
        min_delta (int, optional): Minimum change in the monitored quantity to qualify as an improvement, i.e. an
            absolute change of less than min_delta, will count as no improvement. Defaults to 0.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults
            to 0.
        verbose (int, optional): Verbosity mode.. Defaults to 0.
        compare (str, optional): One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored
            has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing.
            Defaults to 'min'.
        baseline (float, optional): Baseline value for the monitored quantity. Training will stop if the model doesn't
            show improvement over the baseline. Defaults to None.
        restore_best_weights (bool, optional): Whether to restore model weights from the epoch with the best value of
            the monitored quantity. If False, the model weights obtained at the last step of training are used.
            Defaults to False.
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
    """
    def __init__(self,
                 monitor="loss",
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 compare='min',
                 baseline=None,
                 restore_best_weights=False,
                 mode='eval'):
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
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        if compare == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def on_begin(self, state):
        self.wait = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, state):
        current = state.get(self.monitored_key, None)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = dict(map(lambda x: (x[0], x[1].get_weights()), self.network.model.items()))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.network.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('FastEstimator-EarlyStopping: Restoring best model weights')
                    for name, model in self.network.model.items():
                        model.set_weights(self.best_weights[name])
                print("FastEstimator-EarlyStopping: '{}' triggered an early stop. Its best value was {} at epoch {}\
                      ".format(self.monitored_key, self.best, state['epoch'] - self.wait))
