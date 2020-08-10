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
from typing import TypeVar

import tensorflow as tf
import torch

from fastestimator.backend.cast import cast
from fastestimator.backend.clip_by_value import clip_by_value
from fastestimator.backend.reduce_mean import reduce_mean

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def hinge(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the hinge loss between two tensors.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[-1,1,1,-1], [1,1,1,1], [-1,-1,1,-1], [1,-1,-1,-1]])
    pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,-0.2,0.0,-0.7], [0.0,0.15,0.8,0.05], [1.0,-1.0,-1.0,-1.0]])
    b = fe.backend.hinge(y_pred=pred, y_true=true)  # [0.8  1.2  0.85 0.  ]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[-1,1,1,-1], [1,1,1,1], [-1,-1,1,-1], [1,-1,-1,-1]])
    pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,-0.2,0.0,-0.7], [0.0,0.15,0.8,0.05], [1.0,-1.0,-1.0,-1.0]])
    b = fe.backend.hinge(y_pred=pred, y_true=true)  # [0.8  1.2  0.85 0.  ]
    ```

    Args:
        y_true: Ground truth class labels which should take values of 1 or -1.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.

    Returns:
        The hinge loss between `y_true` and `y_pred`

    Raises:
        AssertionError: If `y_true` and `y_pred` have mismatched shapes or disparate types.
        ValueError: If `y_pred` is an unacceptable data type.
    """
    assert y_pred.shape == y_true.shape, \
        f"Hinge loss requires y_true and y_pred to have the same shape, but found {y_true.shape} and {y_pred.shape}"
    y_true = cast(y_true, 'float32')
    return reduce_mean(clip_by_value(1.0 - y_true * y_pred, min_value=0), axis=-1)
