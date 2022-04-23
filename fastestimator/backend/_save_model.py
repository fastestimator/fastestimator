# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import Optional, Union

import tensorflow as tf
import tensorflow_addons as tfa
import torch

from fastestimator.backend._get_lr import get_lr


def save_model(model: Union[tf.keras.Model, torch.nn.Module],
               save_dir: str,
               model_name: Optional[str] = None,
               save_optimizer: bool = False,
               save_architecture: bool = False) -> str:
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
        model_name: The name of the model (used for naming the weights file). If None, model.model_name will be used.
        save_optimizer: Whether to save optimizer. If True, optimizer will be saved in a separate file at same folder.
        save_architecture: Whether to also save the entire model architecture so that the model can be reloaded without
            needing access to the code which generated it. This option is only available for TensorFlow models.

    Returns:
        The saved model path.

    Raises:
        ValueError: If `model` is an unacceptable data type, of if a user tries to save architecture of a PyTorch model.
    """
    assert hasattr(model, "fe_compiled") and model.fe_compiled, "model must be built by fe.build"
    if model_name is None:
        model_name = model.model_name
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, tf.keras.Model):
        model_path = os.path.join(save_dir, "{}.h5".format(model_name))
        model.save_weights(model_path)
        if save_architecture:
            model.save(filepath=os.path.join(save_dir, model_name), include_optimizer=save_optimizer)
        if save_optimizer:
            assert model.current_optimizer, "optimizer does not exist"
            optimizer_path = os.path.join(save_dir, "{}_opt.pkl".format(model_name))
            with open(optimizer_path, 'wb') as f:
                saved_data = {'weights': model.current_optimizer.get_weights(), 'lr': get_lr(model)}
                if isinstance(model.current_optimizer, tfa.optimizers.DecoupledWeightDecayExtension) or hasattr(
                        model.current_optimizer, "inner_optimizer") and isinstance(
                            model.current_optimizer.inner_optimizer, tfa.optimizers.DecoupledWeightDecayExtension):
                    saved_data['weight_decay'] = tf.keras.backend.get_value(model.current_optimizer.weight_decay)
                pickle.dump(saved_data, f)
        return model_path
    elif isinstance(model, torch.nn.Module):
        model_path = os.path.join(save_dir, "{}.pt".format(model_name))
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        if save_architecture:
            raise ValueError("Sorry, architecture saving is not currently enabled for PyTorch")
        if save_optimizer:
            assert model.current_optimizer, "optimizer does not exist"
            optimizer_path = os.path.join(save_dir, "{}_opt.pt".format(model_name))
            torch.save(model.current_optimizer.state_dict(), optimizer_path)
        return model_path
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
