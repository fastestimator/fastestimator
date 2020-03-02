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
from typing import Union

import tensorflow as tf
import torch

from fastestimator.backend.save_model import save_model
from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class ModelSaver(Trace):
    """save model weights based on epoch frequency during training

    Args:
        model: model instance
        save_dir: folder path to save model
        frequency: model saving frequency in epoch(s). Defaults to 1.
    """
    def __init__(self, model: Union[tf.keras.Model, torch.nn.Module], save_dir: str, frequency: int = 1):
        super().__init__(mode="train")
        self.model = model
        self.save_dir = save_dir
        self.frequency = frequency

    def on_epoch_end(self, data: Data):
        if self.save_dir and self.system.epoch_idx % self.frequency == 0:
            model_name = "epoch_{}".format(self.system.epoch_idx)
            save_model(self.model, self.save_dir, model_name)
