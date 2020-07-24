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
from typing import Any, Dict, List, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.op.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class MixUpBatch(TensorOp):
    """ This class should be used in conjunction with MixUpLoss to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412)

    Args:
        inputs: key of the input to be mixed up
        outputs: key to store the mixed-up input
        mode: what mode to execute in. Probably 'train'
        alpha: the alpha value defining the beta distribution to be drawn from during training
        sharedbeta: Sample a single beta for a batch Or single beta for each image
    """
    def __init__(self, inputs: Union[str, List[str]] = None,
                 outputs: Union[str, List[str]] = None,
                 mode: str = None,
                 alpha: float = 1.0,
                 sharedbeta: bool = True,
                 framework: str = 'tf'):
        assert alpha > 0, "Mixup alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha
        self.beta = None
        self.sharedbeta = sharedbeta
        self.build(framework)

    def build(self, framework: str) -> None:
        if framework == 'tf':
            self.alpha = tf.constant(self.alpha)
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
        elif framework == 'torch':
            self.alpha = torch.tensor(self.alpha)
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        """ Forward method to perform mixup batch augmentation

        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.

        Returns:
            Mixed-up batch data
        """
        iterdata = data if isinstance(data, list) else list(data) if isinstance(data, tuple) else [data]

        if self.sharedbeta:
            lam = self.beta.sample()
        else:
            lam = self.beta.sample(sample_shape=data.shape[0])
            lam = tf.maximum(lam, (1 - lam))
            lam = tf.reshape(lam, (-1, 1, 1, 1))

        if tf.is_tensor(data):
            mix = [lam * dat + (1.0 - lam) * tf.roll(dat, shift=1, axis=0) for dat in iterdata]
        elif isinstance(data, torch.Tensor):
            mix = [lam * dat + (1.0 - lam) * torch.roll(dat, shifts=1, dims=0) for dat in iterdata]
        else:
            raise ValueError("unrecognized data format: {}".format(type(data)))

        return mix + [lam]
