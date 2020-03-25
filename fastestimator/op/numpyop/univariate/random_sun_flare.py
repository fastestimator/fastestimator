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
from typing import Union, Iterable, Callable, Tuple

from albumentations.augmentations.transforms import RandomSunFlare as RandomSunFlareAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomSunFlare(ImageOnlyAlbumentation):
    """Add a sun flare to the image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            flare_roi: region of the image where flare will appear (x_min, y_min, x_max, y_max). All values should be
                in range [0, 1].
            angle_lower: should be in range [0, `angle_upper`].
            angle_upper: should be in range [`angle_lower`, 1].
            num_flare_circles_lower (int): lower limit for the number of flare circles.
                Should be in range [0, `num_flare_circles_upper`].
            num_flare_circles_upper (int): upper limit for the number of flare circles.
                Should be in range [`num_flare_circles_lower`, inf].
            src_radius: radius of the flare
            src_color ((int, int, int)): color of the flare
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 flare_roi: Tuple[float, float, float, float] = (0, 0, 1, 0.5),
                 angle_lower: float = 0.0,
                 angle_upper: float = 1.0,
                 num_flare_circles_lower: int = 6,
                 num_flare_circles_upper: int = 10,
                 src_radius: int = 400,
                 src_color: Tuple[int, int, int] = (255, 255, 255)):
        super().__init__(
            RandomSunFlareAlb(flare_roi=flare_roi,
                              angle_lower=angle_lower,
                              angle_upper=angle_upper,
                              num_flare_circles_lower=num_flare_circles_lower,
                              num_flare_circles_upper=num_flare_circles_upper,
                              src_radius=src_radius,
                              src_color=src_color,
                              always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
