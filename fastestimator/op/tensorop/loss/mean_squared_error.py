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
from typing import Union, Iterable, Callable, List, TypeVar, Dict, Any

import tensorflow as tf
import torch

from fastestimator.backend.mean_squared_error import mean_squared_error
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class MeanSquaredError(TensorOp):
    """Calculate mean squared error loss, the rest of the keyword argument will be passed to
    tf.losses.MeanSquaredError

    Args:
        y_true: ground truth label key
        y_pred: prediction label key
        inputs: A tuple or list like: [<y_true>, <y_pred>]
        outputs: Where to store the computed loss value (not required under normal use cases)
        mode: 'train', 'eval', 'test', or None
        kwargs: Arguments to be passed along to the tf.losses constructor
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 average_loss: bool = True):
        self.average_loss = average_loss
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        loss = mean_squared_error(y_true=y_true, y_pred=y_pred)
        if self.average_loss:
            loss = reduce_mean(loss)
        return loss
