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
from typing import Iterable, Tuple, Union

from albumentations.augmentations.transforms import MultiplicativeNoise as MultiplicativeNoiseAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class MultiplicativeNoise(ImageOnlyAlbumentation):
    """Multiply an image with random perturbations.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        multiplier: If a single float, the image will be multiplied by this number. If tuple of floats then `multiplier`
            will be in the range [multiplier[0], multiplier[1]).
        per_channel: Whether to sample different multipliers for each channel of the image.
        elementwise: If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 multiplier: Union[float, Tuple[float, float]] = (0.9, 1.1),
                 per_channel: bool = False,
                 elementwise: bool = False):
        super().__init__(
            MultiplicativeNoiseAlb(multiplier=multiplier,
                                   per_channel=per_channel,
                                   elementwise=elementwise,
                                   always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode,
            ds_id=ds_id)
