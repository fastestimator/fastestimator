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
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend._mean_squared_error import mean_squared_error
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class MeanSquaredError(LossOp):
    """Calculate the mean squared error loss between two tensors.

    Args:
        inputs: A tuple or list like: [y_pred, y_true].
        outputs: String key under which to store the computed loss.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        average_loss: Whether to average the element-wise loss after the Loss Op.
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 average_loss: bool = True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.average_loss = average_loss

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        loss = mean_squared_error(y_true=y_true, y_pred=y_pred)
        if self.average_loss:
            loss = reduce_mean(loss)
        return loss
