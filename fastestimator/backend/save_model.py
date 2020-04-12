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
    """Save `model` weights to a specific directory.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.h5' file inside /tmp directory
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.pt' file inside /tmp directory
    ```

    Args:
        model: A neural network instance to save.
        save_dir: Directory into which to write the `model` weights.
        model_name: The name of the model (used for naming the weights file).

    Raises:
        ValueError: If `model` is an unacceptable data type.
    """
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, tf.keras.Model):
        model_path = os.path.join(save_dir, "{}.h5".format(model_name))
        model.save_weights(model_path)
    elif isinstance(model, torch.nn.Module):
        model_path = os.path.join(save_dir, "{}.pt".format(model_name))
        torch.save(model.state_dict(), model_path)
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
    print("FastEstimator-ModelSaver: saved model to {}".format(model_path))
