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

from fastestimator.backend._dice_score import dice_score
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.array)


class Dice(TensorOp):
    """
    Calculate Element-Wise Dice Score.

        Args:
            inputs: Keys of tensors to be averaged.
            outputs: The key under which to save the output.
            mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
                regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
                like "!infer" or "!train".
            ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
                ds_ids except for a particular one, you can pass an argument like "!ds1".
            soft_dice: Whether to square elements. If True, square of elements is added.
            sample_average: Whether to average the element-wise dice score.
            channel_average: Whether to average the channel wise dice score.
            negate: Whether to negate dice score.
            epsilon: A small value to prevent numeric instability in the division.

        Returns:
            The dice loss between `y_pred` and `y_true`. A scalar if `average_sample` is True, else a
            tensor with the shape (Batch).

        Raises:
            AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """

    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 soft_dice: bool = False,
                 sample_average: bool = False,
                 channel_average: bool = False,
                 negate: bool = False,
                 epsilon: float = 1e-6):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.channel_average = channel_average
        self.soft_dice = soft_dice
        self.epsilon = epsilon
        self.sample_average = sample_average
        self.negate = negate

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        dice = dice_score(
            y_pred, y_true, self.soft_dice, self.sample_average, self.channel_average, self.epsilon)
        return -dice if self.negate else dice
