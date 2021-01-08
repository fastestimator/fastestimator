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
from typing import Iterable, Optional, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.util.util import NonContext

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def get_gradient(target: Tensor,
                 sources: Union[Iterable[Tensor], Tensor],
                 higher_order: bool = False,
                 tape: Optional[tf.GradientTape] = None,
                 retain_graph: bool = True) -> Union[Iterable[Tensor], Tensor]:
    """Calculate gradients of a target w.r.t sources.

    This method can be used with TensorFlow tensors:
    ```python
    x = tf.Variable([1.0, 2.0, 3.0])
    with tf.GradientTape(persistent=True) as tape:
        y = x * x

        b = fe.backend.get_gradient(target=y, sources=x, tape=tape)  # [2.0, 4.0, 6.0]
        b = fe.backend.get_gradient(target=b, sources=x, tape=tape)  # None

        b = fe.backend.get_gradient(target=y, sources=x, tape=tape, higher_order=True)  # [2.0, 4.0, 6.0]
        b = fe.backend.get_gradient(target=b, sources=x, tape=tape)  # [2.0, 2.0, 2.0]
    ```

    This method can be used with PyTorch tensors:
    ```python
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x

    b = fe.backend.get_gradient(target=y, sources=x)  # [2.0, 4.0, 6.0]
    b = fe.backend.get_gradient(target=b, sources=x)  # Error - b does not have a backwards function

    b = fe.backend.get_gradient(target=y, sources=x, higher_order=True)  # [2.0, 4.0, 6.0]
    b = fe.backend.get_gradient(target=b, sources=x)  # [2.0, 2.0, 2.0]
    ```

    Args:
        target: The target (final) tensor.
        sources: A sequence of source (initial) tensors.
        higher_order: Whether the gradient will be used for higher order gradients.
        tape: TensorFlow gradient tape. Only needed when using the TensorFlow backend.
        retain_graph: Whether to retain PyTorch graph. Only valid when using the PyTorch backend.

    Returns:
        Gradient(s) of the `target` with respect to the `sources`.

    Raises:
        ValueError: If `target` is an unacceptable data type.
    """
    if tf.is_tensor(target):
        with NonContext() if higher_order else tape.stop_recording():
            gradients = tape.gradient(target, sources)
    elif isinstance(target, torch.Tensor):
        gradients = torch.autograd.grad(target,
                                        sources,
                                        grad_outputs=torch.ones_like(target),
                                        retain_graph=retain_graph,
                                        create_graph=higher_order,
                                        only_inputs=True)

        if isinstance(sources, torch.Tensor):
            #  The behavior table of tf and torch backend
            #  ---------------------------------------------------------------
            #        | case 1                     | case 2                    |
            #  ---------------------------------------------------------------|
            #  tf    | target: tf.Tensor          | target: tf.Tensor         |
            #        | sources: tf.Tensor         | sources: [tf.Tensor]      |
            #        | gradients: tf.Tensor       | gradients: [tf.Tensor]    |
            # ----------------------------------------------------------------|
            #  torch | target: torch.Tensor       | target: tf.Tensor         |
            #        | sources: torch.Tensor      | sources: [tf.Tensor]      |
            #        | gradients: (torch.Tensor,) | gradients: (torch.Tensor,)|
            # ----------------------------------------------------------------
            # In order to make the torch behavior become the same as tf in case 1, need to unwrap the gradients when
            # source is not Iterable.

            gradients = gradients[0]
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(target)))
    return gradients
