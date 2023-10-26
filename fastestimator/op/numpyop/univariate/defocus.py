# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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


from albumentations.augmentations.blur.transforms import Defocus as DefocusAlb

from fastestimator.util.traceability_util import traceable

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation



@traceable()
class Defocus(ImageOnlyAlbumentation):
    """Apply camera Defocus transform.

    Args:
        radius: range for radius of defocusing. If limit is a single int, the range will be [1, limit]
        alias_blur: range for alias_blur of defocusing (sigma of gaussian blur). If limit is a single float,
            the range will be (0, limit).
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 radius: Union[int, Tuple[int, int]] = (3,10),
                 alias_blur: Union[float, Tuple[float, float]] = (0.1,0.5)
                 ):
        super().__init__(DefocusAlb(radius=radius,
                                     alias_blur=alias_blur,
                                     always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)
