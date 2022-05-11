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
from typing import Callable, Dict, List, Optional, Union

import tensorflow as tf
import torch


_ALREADY_GAVE_FE_GRAD_WARNING = False


def update_model(model: Union[tf.keras.Model, torch.nn.Module],
                 gradients: List[Union[tf.Tensor, torch.Tensor]],
                 defer: bool = False,
                 deferred: Optional[Dict[str, List[Callable[[], None]]]] = None) -> None:
    """Update `model` weights based on a given `gradients`.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
    x = tf.ones((3, 28, 28, 1))  # (batch, height, width, channels)
    y = tf.constant((1, 0, 1))
    with tf.GradientTape(persistent=True) as tape:
        pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
        loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
        gradients = fe.backend.get_gradient(target=loss, sources=m.trainable_variables, tape=tape)
        fe.backend.update_model(m, gradients=gradients)
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
    x = torch.ones((3, 1, 28, 28))  # (batch, channels, height, width)
    y = torch.tensor((1, 0, 1))
    pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
    gradients = fe.backend.get_gradient(target=loss,
                                        sources=[x for x in m.parameters() if x.requires_grad])

    fe.backend.update_model(m, gradients=gradients)
    ```

    Args:
        model: A neural network instance to update.
        gradients: A list of tensors to update the models.
        defer: If True, then the model update function will be stored into the `deferred` dictionary rather than
            applied immediately.
        deferred: A dictionary in which model update functions are stored.

    Raises:
        ValueError: If `model` is an unacceptable data type.
        AssertionError: If `model` doesn't have `current_optimizer` attribute
        AssertionError: If Pytorch `model.current_optimizer` doesn't have `scaler` attribute
    """
    assert hasattr(model, "current_optimizer"), ("The model needs to have 'current_optimizer' attribute. Please "
                                                 "instantiate the model with fe.build")

    if isinstance(model, tf.keras.Model):
        variables = model.trainable_variables
        if defer:
            deferred.setdefault(model.model_name,
                                []).append(lambda: model.current_optimizer.apply_gradients(zip(gradients, variables)))
        else:
            model.current_optimizer.apply_gradients(zip(gradients, variables))

    elif isinstance(model, torch.nn.Module):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        for gradient, parameter in zip(gradients, trainable_params):
            if gradient is None:
                global _ALREADY_GAVE_FE_GRAD_WARNING
                if not _ALREADY_GAVE_FE_GRAD_WARNING:
                    print("\033[93m{}\033[00m".format("FastEstimator-Warn: 'None' detected in gradients. Some or all "
                                                      "of your computation graph may not be connected to your loss."))
                    _ALREADY_GAVE_FE_GRAD_WARNING = True
                continue
            if parameter.grad is not None:
                parameter.grad += gradient
            else:
                parameter.grad = gradient.clone()
        if defer:
            # Only need to call once per model since gradients are getting accumulated
            deferred[model.model_name] = [lambda: _torch_step(model.current_optimizer)]
        else:
            _torch_step(model.current_optimizer)

            if deferred:
                deferred.pop(model.model_name)  # Don't need those deferred steps anymore
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))


def _torch_step(optimizer: torch.optim.Optimizer) -> None:
    assert hasattr(optimizer, "scaler"), ("Pytorch optimizers need to have 'scaler' attribute, Please use fe.build to "
                                          "compile the model")
    if optimizer.scaler is None:
        optimizer.step()
    else:
        optimizer.scaler.step(optimizer)
        optimizer.scaler.update()
    optimizer.zero_grad()
