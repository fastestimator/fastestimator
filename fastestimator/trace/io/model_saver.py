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
from typing import Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.trace import Trace
from fastestimator.util import Data


class ModelSaver(Trace):
    """Save trained model in hdf5/pt format.

    Args:
        model: Model instance defined by user (tf.keras.Model or torch.nn.Module)
        model_name (str): Name of FE model.
        save_dir (str): Directory to save the trained models.
        save_freq (int, optional): Number of epochs to save models. Cannot be used with `save_best_only=True`. Defaults
            to 1.
        save_best (bool, str, optional): Best model saving monitor name. If True, the model loss is used. Defaults to
            False.
        save_best_mode (str, optional): Can be `'min'`, `'max'`, or `'auto'`. Defaults to 'max'.

    """
    def __init__(self, model: Union[tf.keras.Model, torch.nn.Module], model_name: str,
                 save_dir: str, save_freq: int = 1, save_best: Union[bool, str]=False, save_best_mode:str='max'):

        if save_best is True:
            save_best = 'accuracy'

        if isinstance(save_best, str):
            super().__init__(inputs=save_best, mode='eval')
        else:
            super().__init__(mode='train')

        if isinstance(model, tf.keras.Model):
            self.ext = '.h5'
        else:
            self.ext = '.pt'

        self.model = model
        self.model_name=model_name
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
        self.model = model

    def on_begin(self, data: Data):
        if self.save_dir:
            self.save_dir = os.path.normpath(self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, data: Data):
        if isinstance(self.save_best, str):
            if self.monitor_op(data[self.save_best], self.best):
                self.best = data[self.save_best]
                self._save_model("{}_best_{}{}".format(self.model_name, self.save_best, self.ext))
                #self._save_model("{}_best_{}_{}.h5".format(self.model_name, self.save_best, str(self.best)))
        elif self.system.epoch_idx % self.save_freq == 0:
            self._save_model("{}_epoch_{}_step_{}{}".format(
                self.model_name, self.system.epoch_idx, self.system.batch_idx, self.ext))

    def _save_model(self, name: str):
        if self.save_dir:
            save_path = os.path.join(self.save_dir, name)
            if isinstance(self.model, tf.keras.Model):
                self.model.save(save_path, include_optimizer=False)
            else:
                torch.save(self.model, save_path)
            print("FastEstimator-ModelSaver: Saving model to {}".format(save_path))
