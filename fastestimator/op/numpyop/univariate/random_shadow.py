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

from albumentations.augmentations.transforms import RandomShadow as RandomShadowAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomShadow(ImageOnlyAlbumentation):
    """Add shadows to an image

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        shadow_roi: Region of the image where shadows will appear (x_min, y_min, x_max, y_max).
            All values should be in range [0, 1].
        num_shadows_lower: Lower limit for the possible number of shadows. Should be in range [0, `num_shadows_upper`].
        num_shadows_upper: Lower limit for the possible number of shadows.
            Should be in range [`num_shadows_lower`, inf].
        shadow_dimension: Number of edges in the shadow polygons.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 shadow_roi: Tuple[float, float, float, float] = (0.0, 0.5, 1.0, 1.0),
                 num_shadows_lower: int = 1,
                 num_shadows_upper: int = 2,
                 shadow_dimension: int = 5):
        print("\033[93m {}\033[00m".format(
            "Warning! RandomShadow does not work with multi-threaded Pipelines. Either do not use this Op or else " +
            "set your Pipeline num_process=0"))
        # TODO - Have pipeline look for bad ops and auto-magically set num_process correctly
        super().__init__(
            RandomShadowAlb(shadow_roi=shadow_roi,
                            num_shadows_lower=num_shadows_lower,
                            num_shadows_upper=num_shadows_upper,
                            shadow_dimension=shadow_dimension,
                            always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode,
            ds_id=ds_id)
