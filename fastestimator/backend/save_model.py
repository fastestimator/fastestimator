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

import tensorflow as tf
import torch


def save_model(model: Union[tf.keras.Model, torch.nn.Module], save_dir: str, model_name: str = "saved_model"):
    """Save tensorflow or pytorch model weights to a specific directory

    Args:
        model : model instance
        save_dir :folder path to save model
        model_name : name of the model without extension
    """
    assert isinstance(model, (tf.keras.Model, torch.nn.Module)), "unsupported model instance type"
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, tf.keras.Model):
        model.save_weights(os.path.join(save_dir, "{}.h5".format(model_name)))

    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "{}.pt".format(model_name)))
        print("saved model to {}".format(os.path.join(save_dir, "{}.h5".format(model_name))))