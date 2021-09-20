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
import tensorflow_probability as tfp
import torch

from fastestimator.backend.maximum import maximum
from fastestimator.backend.reshape import reshape
from fastestimator.backend.roll import roll
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class MixUpBatch(TensorOp):
    """MixUp augmentation for tensors.

    This class should be used in conjunction with MixLoss to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412).

    Args:
        inputs: Key of the input to be mixed up.
        outputs: Key to store the mixed-up outputs.
        mode: What mode to execute in. Probably 'train'.
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        alpha: The alpha value defining the beta distribution to be drawn from during training.
        shared_beta: Sample a single beta for a batch or element wise beta for each image.

    Raises:
        AssertionError: If input arguments are invalid.
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Iterable[str],
                 mode: Union[None, str, Iterable[str]] = 'train',
                 ds_id: Union[None, str, Iterable[str]] = None,
                 alpha: float = 1.0,
                 shared_beta: bool = True):
        assert alpha > 0, "MixUp alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        assert len(self.outputs) == len(self.inputs) + 1, "MixUpBatch requires 1 more output than inputs"
        self.alpha = alpha
        self.beta = None
        self.shared_beta = shared_beta
        self.in_list, self.out_list = True, True

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if framework == 'tf':
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
        elif framework == 'torch':
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        if self.shared_beta:
            lam = self.beta.sample()
        else:
            lam = self.beta.sample(sample_shape=(data[0].shape[0], ))
            shape = [-1] + [1] * (len(data[0].shape) - 1)
            lam = reshape(lam, shape)
        lam = maximum(lam, (1 - lam))
        mix = [lam * elem + (1.0 - lam) * roll(elem, shift=1, axis=0) for elem in data]
        return mix + [lam]
