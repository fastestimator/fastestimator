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


def load_model(model: Union[tf.keras.Model, torch.nn.Module], weights_path: str):
    """Save tensorflow or pytorch model weights to a specific directory

    Args:
        model : model instance
        save_dir :folder path to save model
        model_name : name of the model without extension
    """
    assert isinstance(model, (tf.keras.Model, torch.nn.Module)), "unsupported model instance type"

    if isinstance(model, tf.keras.Model):
        model.load_weights(weights_path)
    else:
        model.load_state_dict(torch.load(weights_path))
    print("Loaded model weights from {}".format(weights_path))
