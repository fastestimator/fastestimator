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


def set_lr(model: Union[tf.keras.Model, torch.nn.Module], lr: float):
    """set the learning rate of a given model

    Args:
        model: model instance
        lr: learning rate to set
    """
    if isinstance(model, tf.keras.Model):
        tf.keras.backend.set_value(model.current_optimizer.lr, lr)
    else:
        for param_group in model.current_optimizer.param_groups:
            param_group['lr'] = lr
