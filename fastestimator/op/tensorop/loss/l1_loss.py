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
from ctypes.wintypes import HACCEL
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import tensorflow as tf
import torch
from sklearn.linear_model import HuberRegressor

from fastestimator.backend.huber import Huber
from fastestimator.backend.l1_loss import l1_loss
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.smooth_l1_loss import Smooth_l1_loss
from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class L1_Loss(LossOp):
    """Calculate the mean squared error loss between two tensors.

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>].
        outputs: String key under which to store the computed loss.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        average_loss: Whether to average the element-wise loss after the Loss Op.
        loss_type: What type of L1 loss. Can either be None (L1 Loss), 'Smooth' (Smooth L1 Loss) or 'Huber' (Huber loss)
        beta: Threshold factor. Needs to be a positive number. dtype: float16 or float32.
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 average_loss: bool = True,
                 loss_type: Union[None, str] = None,
                 beta: Union[None, float] = 1.0):
        self.average_loss = average_loss
        self.loss_type = loss_type
        self.beta = beta
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        if self.loss_type == None:
            loss = l1_loss(y_true=y_true, y_pred=y_pred)
        elif self.loss_type == 'Smooth':
            loss = Smooth_l1_loss(y_true=y_true, y_pred=y_pred, beta=self.beta)
        elif self.loss_type == 'Huber':
            loss = Huber(y_true=y_true, y_pred=y_pred, beta=self.beta)
        else:
            raise ValueError(
                "Unrecognized Loss type. Can either be None (L1 Loss), 'Smooth' (Smooth L1 Loss) or 'Huber' (Huber loss)"
            )
        if self.average_loss:
            loss = reduce_mean(loss)
        return loss
