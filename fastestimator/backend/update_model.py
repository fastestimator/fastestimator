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
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from fastestimator.backend.get_gradient import get_gradient
from fastestimator.backend.reduce_mean import reduce_mean


def update_model(model: Union[tf.keras.Model, torch.nn.Module],
                 loss: Union[None, tf.Tensor, torch.Tensor] = None,
                 gradients: Optional[List[Union[tf.Tensor, torch.Tensor]]] = None,
                 tape: Optional[tf.GradientTape] = None,
                 retain_graph: bool = True,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None,
                 defer: bool = False,
                 deferred: Optional[Dict[str, List[Callable[[], None]]]] = None) -> None:
    """Update `model` weights based on a given `loss`.

    This method can be used with TensorFlow models:
    ```python
    m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
    x = tf.ones((3,28,28,1))  # (batch, height, width, channels)
    y = tf.constant((1, 0, 1))
    with tf.GradientTape(persistent=True) as tape:
        pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
        loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
        fe.backend.update_model(m, loss=loss, tape=tape)
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
    x = torch.ones((3,1,28,28))  # (batch, channels, height, width)
    y = torch.tensor((1, 0, 1))
    pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
    fe.backend.update_model(m, loss=loss)
    ```

    Args:
        model: A neural network instance to update.
        loss: A loss value to compute gradients from, mutually exclusive with `gradients`.
        gradients: A list of tensors to update the models, mutually exclusive with `loss`.
        tape: A TensorFlow GradientTape which was recording when the `loss` was computed (iff using TensorFlow).
        retain_graph: Whether to keep the model graph in memory (applicable only for PyTorch).
        scaler: A PyTorch loss scaler that scales loss when PyTorch mixed precision is used.
        defer: If True, then the model update function will be stored into the `deferred` dictionary rather than
            applied immediately.
        deferred: A dictionary in which model update functions are stored.

    Raises:
        ValueError: If `model` is an unacceptable data type.
        RuntimeError: If attempting to modify a PyTorch model which relied on gradients within a different PyTorch model
            which has in turn already undergone a non-deferred update.
    """
    if loss is not None:
        loss = reduce_mean(loss)
    if isinstance(model, tf.keras.Model):
        if loss is not None:
            # scale up loss for mixed precision training to avoid underflow
            if isinstance(model.current_optimizer, mixed_precision.LossScaleOptimizer):
                loss = model.current_optimizer.get_scaled_loss(loss)
            # for multi-gpu training, the gradient will be combined by sum, normalize the loss
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.MirroredStrategy):
                loss = loss / strategy.num_replicas_in_sync
            gradients = get_gradient(loss, model.trainable_variables, tape=tape)
        with tape.stop_recording():
            # scale down gradient to balance scale-up loss
            if isinstance(model.current_optimizer, mixed_precision.LossScaleOptimizer):
                gradients = model.current_optimizer.get_unscaled_gradients(gradients)
            if defer:
                deferred.setdefault(model.model_name, []).append(
                    lambda: model.current_optimizer.apply_gradients(zip(gradients, model.trainable_variables)))
            else:
                model.current_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    elif isinstance(model, torch.nn.Module):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # scale up loss for mixed precision training to avoid underflow
        if scaler is not None:
            loss = scaler.scale(loss)
        if loss is not None:
            try:
                gradients = get_gradient(loss, trainable_params, retain_graph=retain_graph)
            except RuntimeError as err:
                if err.args and isinstance(err.args[0], str) and err.args[0].startswith(
                        'one of the variables needed for gradient computation has been modified by an inplace operation'
                ):
                    raise RuntimeError(
                        "When computing gradients for '{}', some variables it relied on during the forward pass had"
                        " been updated. Consider setting defer=True in earlier UpdateOps related to models which "
                        "interact with this one.".format(model.model_name))
                raise err
        for gradient, parameter in zip(gradients, trainable_params):
            if parameter.grad is not None:
                parameter.grad += gradient
            else:
                parameter.grad = gradient.clone()
        if defer:
            # Only need to call once per model since gradients are getting accumulated
            deferred[model.model_name] = [lambda: _torch_step(model.current_optimizer, scaler)]
        else:
            _torch_step(model.current_optimizer, scaler)
            deferred.pop(model.model_name, None)  # Don't need those deferred steps anymore
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))


def _torch_step(optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> None:
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
    optimizer.zero_grad()
