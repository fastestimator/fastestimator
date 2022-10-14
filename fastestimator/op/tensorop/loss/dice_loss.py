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
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._convert_tensor_precision import convert_tensor_precision
from fastestimator.backend._dice_score import dice_score
from fastestimator.backend._to_tensor import to_tensor
from fastestimator.op.tensorop.loss.loss import LossOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.array)


class DiceLoss(LossOp):
    """Calculate Dice Loss.

    Args:
        inputs: A tuple or list of keys representing prediction and ground truth, like: ("y_pred", "y_true").
        outputs: The key under which to save the output.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        soft_dice: Whether to square elements in the denominator.
        average_loss: Whether to average the element-wise loss after the Loss Op.
        channel_average: Whether to average the dice score along the channel dimension.
        channel_weights: Dictionary mapping channel indices to a weight for weighting the loss function. Useful when you
            need to pay more attention to a particular channel.
        epsilon: A small value to prevent numeric instability in the division.

    Returns:
        The dice loss between `y_pred` and `y_true`. A scalar if `sample_average` and `channel_average` are True,
        otherwise a tensor.

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """

    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 soft_dice: bool = False,
                 average_loss: bool = True,
                 channel_average: bool = True,
                 channel_weights: Optional[Dict[int, float]] = None,
                 epsilon: float = 1e-6):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id, average_loss=average_loss)
        self.channel_average = channel_average
        self.soft_dice = soft_dice
        self.epsilon = epsilon
        if channel_weights:
            assert isinstance(channel_weights, dict), \
                "channel_weights should be a dictionary or have None value, got {}".format(type(channel_weights))
            assert all(isinstance(key, int) for key in channel_weights.keys()), \
                "Please ensure that the keys of the class_weight dictionary are of type: int"
            assert all(isinstance(value, float) for value in channel_weights.values()), \
                "Please ensure that the values of the class_weight dictionary are of type: float"
        self.weights = None
        if channel_weights is not None:
            self.weights = np.ones((1, max(channel_weights.keys()) + 1), dtype='float32')
            for channel, weight in channel_weights.items():
                self.weights[0, channel] = weight

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if framework == 'tf':
            if self.weights is not None:
                self.weights = convert_tensor_precision(to_tensor(self.weights, 'tf'))
        elif framework == 'torch':
            if self.weights is not None:
                self.weights = convert_tensor_precision(to_tensor(self.weights, 'torch'))
                self.weights.to(device)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        dice = dice_score(y_pred=y_pred,
                          y_true=y_true,
                          soft_dice=self.soft_dice,
                          sample_average=self.average_loss,
                          channel_average=self.channel_average,
                          channel_weights=self.weights,
                          epsilon=self.epsilon)
        return -dice
