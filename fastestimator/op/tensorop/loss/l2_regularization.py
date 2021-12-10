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
import torch
import tensorflow as tf

from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.backend.l2_regularization import l2_regularization
from fastestimator.util.traceability_util import traceable
from typing import Any, Dict, List, Tuple, TypeVar, Union, Iterable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class L2Regularizaton(LossOp):
    """Calculate L2 Regularization Loss.

    Args:
        inputs: String key representing input loss.
        outputs: String key under which to store the computed loss value.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".

    Raises:
        AssertionError: If `class_weights` or it's keys and values are of unacceptable data types.
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 mode: Union[None, str, Iterable[str]] = None,
                 beta: float = 0.01):
        super().__init__(inputs=inputs, outputs=outputs, mode = mode)

        self.model = model
        self.beta = beta

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Tensor:
        '''
        For pytorch: param.norm(2) is similar to `param.pow(2).sum().sqrt()`
        For tensorflow: tf.nn.l2_loss(w) is similar to `tf.reduce_sum(tf.pow(w,2)) / 2`
        '''
        loss = data
        total_loss = l2_regularization(loss, self.model, self.beta) + loss
        
        return total_loss
