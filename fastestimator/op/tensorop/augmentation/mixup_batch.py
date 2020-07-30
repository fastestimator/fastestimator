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
from typing import Any, Dict, Iterable, List, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

import fastestimator as fe
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class MixUpBatch(TensorOp):
    """ MixUp augmentation for tensors.

    This class should be used in conjunction with MixLoss to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412)

    Args:
        inputs: Key of the input to be mixed up.
        outputs: Key to store the mixed-up outputs.
        mode: What mode to execute in. Probably 'train'.
        alpha: The alpha value defining the beta distribution to be drawn from during training.
        shared_beta: Sample a single beta for a batch or element wise beta for each image.
    """
    def __init__(self, inputs: Union[str, List[str]],
                 outputs: List[str],
                 mode: Union[None, str, Iterable[str]] = 'train',
                 alpha: float = 1.0,
                 shared_beta: bool = False):
        assert alpha > 0, "MixUp alpha value must be greater than zero"
        assert len(outputs) > 1, "Outputs should have at least two arguments"

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha
        self.beta = None
        self.shared_beta = shared_beta

    def build(self, framework: str) -> None:
        if framework == 'tf':
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
        elif framework == 'torch':
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        """ Forward method to perform MixUpBatch augmentation

        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.

        Returns:
            Mixed-up batch data
        """
        if self.shared_beta:
            lam = self.beta.sample()
        else:
            lam = self.beta.sample(sample_shape=data.shape[0])
            lam = fe.backend.maximum(lam, (1 - lam))
            shape = list(data.shape)
            shape[0] = -1
            shape[1:] = [1] * (len(shape) - 1)
            lam = fe.backend.reshape(lam, shape)

        mix = lam * data + (1.0 - lam) * fe.backend.roll(data, shift=1, axis=0)

        return mix, lam
