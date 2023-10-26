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
from typing import Any, Callable, Iterable, Tuple, Union

from albumentations.augmentations import PixelDistributionAdaptation as PixelDistributionAdaptationAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class PixelDistributionAdaptation(ImageOnlyAlbumentation):
    """Naive and quick pixel-level domain adaptation.

    It provides pixel-level domain adaptation by fitting a simple transform (such as PCA, StandardScaler or
    MinMaxScaler) on both the original and reference image, transforming the original image with the transform
    trained on this image, and then performing an inverse transformation using the transform fitted on
    the reference image.

    Args:
        reference_images: Sequence of objects that will be converted to images by read_fn. Can either be path
            of images or numpy arrays (depends upon read_fn).
        blend_ratio: Tuple of min and max blend ratio. Matched image will be blended with original with
            random blend factor for increased diversity of generated images.
        read_fn: User-defined function to read image, tensor or numpy array. Function should get an element
            of reference_images
        transform_type: type of transform; "pca", "standard", "minmax" are allowed.
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
                 reference_images: Union[Any, Iterable[Any]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 blend_ratio: Tuple[float, float] = (0.25,1.0),
                 read_fn: Callable = lambda x: x, # for reading tensor to numpy array
                 transform_type: str = 'pca'
                 ):
        super().__init__(PixelDistributionAdaptationAlb(reference_images=reference_images,
                                blend_ratio=blend_ratio,
                                read_fn=read_fn,
                                transform_type=transform_type,
                                always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)

