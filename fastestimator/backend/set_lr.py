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
    """Set the learning rate of a given `model`.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")  # m.optimizer.lr == 0.001
    fe.backend.set_lr(m, lr=0.8)  # m.optimizer.lr == 0.8
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")  # m.optimizer.param_groups[-1]['lr'] == 0.001
    fe.backend.set_lr(m, lr=0.8)  # m.optimizer.param_groups[-1]['lr'] == 0.8
    ```

    Args:
        model: A neural network instance to modify.
        lr: The learning rate to assign to the `model`.

    Raises:
        ValueError: If `model` is an unacceptable data type.
    """
    if isinstance(model, tf.keras.Model):
        tf.keras.backend.set_value(model.current_optimizer.lr, lr)
    elif isinstance(model, torch.nn.Module):
        for param_group in model.current_optimizer.param_groups:
            param_group['lr'] = lr
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
