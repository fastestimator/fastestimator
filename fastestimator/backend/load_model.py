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
    """Load saved weights for a given model.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="tmp")
    fe.backend.load_model(m, weights_path="tmp/saved_model.h5")
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="tmp")
    fe.backend.load_model(m, weights_path="tmp/saved_model.pt")
    ```

    Args:
        model: A neural network instance to load.
        weights_path: Path to the `model` weights. 
    """
    if isinstance(model, tf.keras.Model):
        model.load_weights(weights_path)
    elif isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(weights_path))
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
    print("Loaded model weights from {}".format(weights_path))
