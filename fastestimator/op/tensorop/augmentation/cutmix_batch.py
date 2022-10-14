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

from fastestimator.backend._cast import cast
from fastestimator.backend._clip_by_value import clip_by_value
from fastestimator.backend._roll import roll
from fastestimator.backend._get_image_dims import get_image_dims
from fastestimator.backend._maximum import maximum
from fastestimator.backend._tensor_round import tensor_round
from fastestimator.backend._tensor_sqrt import tensor_sqrt
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class CutMixBatch(TensorOp):
    """This class performs cutmix augmentation on a batch of tensors.

    In this augmentation technique patches are cut and pasted among training images where the ground truth labels are
    also mixed proportionally to the area of the patches. This class helps to reduce over-fitting, perform object
    detection, and against adversarial
    attacks (https://arxiv.org/pdf/1905.04899.pdf).

    Args:
        inputs: Keys of the image batch and label batch to be cut-mixed.
        outputs: Keys under which to store the cut-mixed images and cut-mixed label.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        alpha: The alpha value defining the beta distribution to be drawn from during training which controls the
            combination ratio between image pairs.

    Raises:
        AssertionError: If the provided inputs are invalid.
    """
    def __init__(self,
                 inputs: Iterable[str],
                 outputs: Iterable[str],
                 mode: Union[None, str, Iterable[str]] = 'train',
                 ds_id: Union[None, str, Iterable[str]] = None,
                 alpha: Union[float, Tensor] = 1.0) -> None:
        assert alpha > 0, "Alpha value must be greater than zero"
        assert len(inputs) == 2, "Cut-Mix must have exactly 2 inputs"
        assert len(outputs) == 2, "Cut-Mix must have exactly 2 outputs"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.alpha = alpha
        self.beta = None
        self.uniform = None

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if framework == 'tf':
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
            self.uniform = tfp.distributions.Uniform()
        elif framework == 'torch':
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
            self.uniform = torch.distributions.uniform.Uniform(low=0, high=1)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    @staticmethod
    def _get_patch_coordinates(tensor: Tensor, x: Tensor, y: Tensor,
                               lam: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly cut the patches from input images.

        If patches are going to be pasted in other image, combination ratio between two images is defined by `lam`.
        Cropping region indicates where to drop out from the image and `cut_x` & `cut_y` are used to calculate cropping
        region whose aspect ratio is proportional to the original image.

        Args:
            tensor: The input value.
            lam: Combination ratio between two images. Larger the lambda value is smaller the patch would be. A
                scalar tensor containing value between 0 and 1.
            x: X-coordinate in image from which patch needs to be cropped. A scalar tensor containing value between 0
                and 1 which in turn is transformed in the range of image width.
            y: Y-coordinate in image from which patch needs to be cropped. A scalar tensor containing value between 0
                and 1 which in turn is transformed in the range of image height.

        Returns:
            The X and Y coordinates of the cropped patch along with width and height.
        """
        _, img_height, img_width = get_image_dims(tensor)

        if tf.is_tensor(tensor):
            ht = cast(img_height, "float32")
            wd = cast(img_width, "float32")
        else:
            ht = img_height
            wd = img_width

        cut_x = wd * x
        cut_y = ht * y
        cut_w = wd * tensor_sqrt(1 - lam)
        cut_h = ht * tensor_sqrt(1 - lam)
        bbox_x1 = cast(tensor_round(clip_by_value(cut_x - cut_w / 2, min_value=0)), "int32")
        bbox_x2 = cast(tensor_round(clip_by_value(cut_x + cut_w / 2, max_value=wd)), "int32")
        bbox_y1 = cast(tensor_round(clip_by_value(cut_y - cut_h / 2, min_value=0)), "int32")
        bbox_y2 = cast(tensor_round(clip_by_value(cut_y + cut_h / 2, max_value=ht)), "int32")
        return bbox_x1, bbox_x2, bbox_y1, bbox_y2, img_width, img_height

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        x, y = data
        lam = self.beta.sample()
        lam = maximum(lam, (1 - lam))
        cut_x = self.uniform.sample()
        cut_y = self.uniform.sample()
        bbox_x1, bbox_x2, bbox_y1, bbox_y2, width, height = self._get_patch_coordinates(x, cut_x, cut_y, lam=lam)
        if tf.is_tensor(x):
            rolled_x = roll(x, shift=1, axis=0)
            patches = rolled_x[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :] - x[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :]
            patches = tf.pad(patches, [[0, 0], [bbox_y1, height - bbox_y2], [bbox_x1, width - bbox_x2], [0, 0]],
                             mode="CONSTANT",
                             constant_values=0)
            x = x + patches
        else:
            rolled_x = roll(x, shift=1, axis=0)
            x[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2] = rolled_x[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        # adjust lambda to match pixel ratio
        lam = 1 - cast(((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)), dtype=y) / cast((width * height), dtype=y)
        rolled_y = roll(y, shift=1, axis=0) * (1. - lam)
        mixed_y = (y * lam) + rolled_y
        return x, mixed_y
