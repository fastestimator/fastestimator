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

import tensorflow as tf
import torch

from fastestimator.backend._focal_loss import focal_loss
from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class FocalLoss(LossOp):
    """Calculate Focal Loss.

    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>].
        outputs: String key under which to store the computed loss value.
        alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 to ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        sample_reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        shape_reduction:
                 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged across classes.
                 'sum': The output will be summed across classes.
        from_logits: Whether y_pred is logits (without sigmoid).
        normalize: Whether to normalize focal loss along samples based on number of positive classes per samples.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an
            argument like "!infer" or "!train".
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 sample_reduction: str = 'mean',
                 shape_reduction: str = 'sum',
                 from_logits: bool = False,
                 normalize: bool = True,
                 mode: Union[None, str, Iterable[str]] = "!infer"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.gamma = gamma
        self.alpha = alpha
        self.sample_reduction = sample_reduction
        self.shape_reduction = shape_reduction
        self.from_logits = from_logits
        self.normalize = normalize

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        return focal_loss(y_true,
                          y_pred,
                          gamma=self.gamma,
                          alpha=self.alpha,
                          from_logits=self.from_logits,
                          sample_reduction=self.sample_reduction,
                          shape_reduction=self.shape_reduction,
                          normalize=self.normalize)
