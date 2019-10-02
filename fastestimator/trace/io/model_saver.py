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

import numpy as np

from fastestimator.trace import Trace


class ModelSaver(Trace):
    """Save trained model in hdf5 format.

    Args:
        model_name (str): Name of FE model.
        save_dir (str): Directory to save the trained models.
        save_best (bool, str, optional): Best model saving monitor name. If True, the model loss is used. Defaults to
            False.
        save_best_mode (str, optional): Can be `'min'`, `'max'`, or `'auto'`. Defaults to 'min'.
        save_freq (int, optional): Number of epochs to save models. Cannot be used with `save_best_only=True`. Defaults
            to 1.
    """
    def __init__(self, model_name, save_dir, save_best=False, save_best_mode='min', save_freq=1):
        if isinstance(save_best, str):
            super().__init__(inputs=save_best)
        else:
            super().__init__()
        self.model_name = model_name
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_best_mode = save_best_mode
        self.save_freq = save_freq
        assert isinstance(self.save_freq, int), "save_freq must be integer"
        if self.save_best_mode == "min":
            self.best = np.Inf
            self.monitor_op = np.less
        elif self.save_best_mode == "max":
            self.best = -np.Inf
            self.monitor_op = np.greater
        else:
            raise ValueError("save_best_mode must be either 'min' or 'max'")
        self.model = None

    def on_begin(self, state):
        if self.save_dir:
            self.save_dir = os.path.normpath(self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)
        self.model = self.network.model[self.model_name]
        if self.save_best is True:
            self.save_best = self.model.loss_name

    def on_epoch_end(self, state):
        if self.save_best:
            if state["mode"] == "eval" and self.monitor_op(state[self.save_best], self.best):
                self.best = state[self.save_best]
                self._save_model("{}_best_{}.h5".format(self.model_name, self.save_best))
        elif state["mode"] == "train" and state["epoch"] % self.save_freq == 0:
            self._save_model("{}_epoch_{}_step_{}.h5".format(self.model_name, state['epoch'], state['train_step']))

    def _save_model(self, name):
        if self.save_dir:
            save_path = os.path.join(self.save_dir, name)
            self.model.save(save_path, include_optimizer=False)
            print("FastEstimator-ModelSaver: Saving model to {}".format(save_path))
