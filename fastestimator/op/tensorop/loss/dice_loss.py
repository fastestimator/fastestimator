# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.dice_loss import dice_loss
from fastestimator.op.tensorop.loss.loss import LossOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.array)


class DiceLoss(LossOp):
    """
    Calculate Element-Wise Dice loss.

        Args:
            y_pred: Prediction with a shape like (Batch, C). dtype: float32 or float16.
            y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
            log_loss: Whether to commute log loss. If True, log loss will be applied instead of dice loss.
            soft_dice_loss: Whether to square elements. If True, square of elements is added.
            average_sample_loss: Whether to average the element-wise loss.
            channel_average: Whether to average the channel wise loss.

        Returns:
            The dice loss between `y_pred` and `y_true`. A scalar if `average_sample_loss` is True, else a
            tensor with the shape (Batch).

        Raises:
            AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """

    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 log_loss: bool = False,
                 soft_dice_loss: bool = False,
                 average_sample_loss: bool = False,
                 channel_average: bool = False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode,
                         ds_id=ds_id, average_loss=average_sample_loss)
        self.channel_average = channel_average
        self.log_loss = log_loss
        self.soft_dice_loss = soft_dice_loss

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        return dice_loss(y_pred, y_true, self.log_loss, self.soft_dice_loss, self.average_loss, self.channel_average)
