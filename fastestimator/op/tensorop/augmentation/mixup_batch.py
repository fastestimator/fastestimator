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
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.backend._get_shape import get_shape
from fastestimator.backend._maximum import maximum
from fastestimator.backend._reshape import reshape
from fastestimator.backend._roll import roll
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class MixUpBatch(TensorOp):
    """MixUp augmentation for tensors.

    This class helps to reduce over-fitting, stabilize GAN training, and against adversarial attacks
    (https://arxiv.org/abs/1710.09412).

    Args:
        inputs: Keys of the image batch and label batch to be mixed up.
        outputs: Keys under which to store the mixed up images and mixed up label.
        mode: What mode to execute in. Probably 'train'.
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        alpha: The alpha value defining the beta distribution to be drawn from during training.
        shared_beta: Sample a single beta for a batch or element wise beta for each image.

    Raises:
        AssertionError: If input arguments are invalid.
    """
    def __init__(self,
                 inputs: Iterable[str],
                 outputs: Iterable[str],
                 mode: Union[None, str, Iterable[str]] = 'train',
                 ds_id: Union[None, str, Iterable[str]] = None,
                 alpha: float = 1.0,
                 shared_beta: bool = False):
        assert alpha > 0, "MixUp alpha value must be greater than zero"
        assert len(inputs) == 2, "MixUp must have exactly 2 inputs"
        assert len(outputs) == 2, "MixUp must have exactly 2 outputs"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.alpha = alpha
        self.beta = None
        self.shared_beta = shared_beta

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if framework == 'tf':
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
        elif framework == 'torch':
            self.beta = torch.distributions.beta.Beta(
                torch.tensor([self.alpha]).to(device), torch.tensor([self.alpha]).to(device))
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        x, y = data

        if self.shared_beta:
            lam = self.beta.sample()
            lam_y = lam
        else:
            shp = get_shape(x)
            lam = self.beta.sample(sample_shape=(shp[0], ))
            shape = [-1] + [1] * (len(shp) - 1)
            lam = reshape(lam, shape)
            # Shape for labels
            shape_y = [-1] + [1] * (len(y.shape) - 1)
            lam_y = reshape(lam, shape_y)

        # To ensure we are not merging the same images
        lam = maximum(lam, (1 - lam))
        # Merge Images
        rolled_x = roll(x, shift=1, axis=0) * (1. - lam)
        mixed_x = (x * lam) + rolled_x
        # Merge Labels
        rolled_y = roll(y, shift=1, axis=0) * (1. - lam_y)
        mixed_y = (y * lam_y) + rolled_y

        return mixed_x, mixed_y
