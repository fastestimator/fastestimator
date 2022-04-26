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
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend._clip_by_value import clip_by_value
from fastestimator.backend._get_gradient import get_gradient
from fastestimator.backend._reduce_max import reduce_max
from fastestimator.backend._reduce_min import reduce_min
from fastestimator.backend._sign import sign
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class FGSM(TensorOp):
    """Create an adversarial sample from input data using the Fast Gradient Sign Method.

    See https://arxiv.org/abs/1412.6572 for an explanation of adversarial attacks.

    Args:
        data: Key of the input to be attacked.
        loss: Key of the loss value to use for gradient computation.
        outputs: The key under which to save the output.
        epsilon: The strength of the perturbation to use in the attack.
        clip_low: a minimum value to clip the output by (defaults to min value of data when set to None).
        clip_high: a maximum value to clip the output by (defaults to max value of data when set to None).
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 data: str,
                 loss: str,
                 outputs: str,
                 epsilon: float = 0.01,
                 clip_low: Optional[float] = None,
                 clip_high: Optional[float] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=[data, loss], outputs=outputs, mode=mode, ds_id=ds_id)
        self.epsilon = epsilon
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.retain_graph = True

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        data, loss = data
        grad = get_gradient(target=loss, sources=data, tape=state['tape'], retain_graph=self.retain_graph)
        adverse_data = clip_by_value(data + self.epsilon * sign(grad),
                                     min_value=self.clip_low or reduce_min(data),
                                     max_value=self.clip_high or reduce_max(data))
        return adverse_data
