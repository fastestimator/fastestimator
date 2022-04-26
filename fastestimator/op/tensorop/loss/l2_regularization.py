# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from typing import Any, Dict, Iterable, List, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend._l2_regularization import l2_regularization
from fastestimator.op.tensorop.tensorop import TensorOp

from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class L2Regularizaton(TensorOp):
    """Calculate L2 Regularization Loss.

    Args:
        inputs: String key representing input loss.
        outputs: String key under which to store the computed loss value.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        model: A tensorflow or pytorch model
        beta: The multiplicative factor, to weight the l2 regularization loss with the input loss
    """
    def __init__(self,
                 inputs: str,
                 outputs: str,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 mode: Union[None, str, Iterable[str]] = None,
                 beta: float = 0.01):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

        self.model = model
        self.beta = beta

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Tensor:
        loss = data
        total_loss = l2_regularization(self.model, self.beta) + loss
        return total_loss
