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
import pickle
from typing import Union

import tensorflow as tf
import torch


def load_model(model: Union[tf.keras.Model, torch.nn.Module], weights_path: str, load_optimizer: bool = False):
    """Load saved weights for a given model.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="tmp", model_name="test")
    fe.backend.load_model(m, weights_path="tmp/test.h5")
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
    fe.backend.save_model(m, save_dir="tmp", model_name="test")
    fe.backend.load_model(m, weights_path="tmp/test.pt")
    ```

    Args:
        model: A neural network instance to load.
        weights_path: Path to the `model` weights.
        load_optimizer: Whether to load optimizer. If True, then it will load <weights_opt> file in the path.

    Raises:
        ValueError: If `model` is an unacceptable data type.
    """
    assert hasattr(model, "fe_compiled") and model.fe_compiled, "model must be built by fe.build"
    if isinstance(model, tf.keras.Model):
        model.load_weights(weights_path)
        if load_optimizer:
            assert model.current_optimizer, "optimizer does not exist"
            optimizer_path = "{}_opt.pkl".format(os.path.splitext(weights_path)[0])
            assert os.path.exists(optimizer_path), "cannot find optimizer path: {}".format(optimizer_path)
            with open(optimizer_path, 'rb') as f:
                weight_values = pickle.load(f)
            model.current_optimizer.set_weights(weight_values)
    elif isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(weights_path))
        if load_optimizer:
            assert model.current_optimizer, "optimizer does not exist"
            optimizer_path = "{}_opt.pt".format(os.path.splitext(weights_path)[0])
            assert os.path.exists(optimizer_path), "cannot find optimizer path: {}".format(optimizer_path)
            model.current_optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
