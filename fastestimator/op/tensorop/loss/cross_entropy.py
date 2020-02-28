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
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.cross_entropy import cross_entropy
from fastestimator.op.op import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class CrossEntropy(TensorOp):
    """Calculate Element-Wise CrossEntropy(binary, categorical or sparse categorical)

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>]
        outputs: key to store the computed loss value
        mode: 'train', 'eval' or None
        from_logits: whether y_pred is logits (without softmax). Defaults to False.
        average_loss: whether to average the element-wise loss after the Loss Op
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 from_logits: bool = False,
                 average_loss: bool = True):
        self.from_logits = from_logits
        self.average_loss = average_loss
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        loss = cross_entropy(y_pred, y_true, from_logits=self.from_logits, average_loss=self.average_loss)
        return loss
