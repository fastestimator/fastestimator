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


def get_lr(model: Union[tf.keras.Model, torch.nn.Module]) -> float:
    """get the learning rate of a given model

    Args:
        model: model instance
    Return:
        lr: learning rate
    """
    if isinstance(model, tf.keras.Model):
        lr = tf.keras.backend.get_value(model.current_optimizer.lr)
    else:
        lr = model.current_optimizer.param_groups[0]['lr']
    return lr
