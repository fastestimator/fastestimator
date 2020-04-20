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
from typing import Callable, Iterable, Union

from albumentations.augmentations.transforms import ImageCompression as ImgCmpAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class ImageCompression(ImageOnlyAlbumentation):
    """Decrease compression of an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        quality_lower: Lower bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper: Upper bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type: should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 quality_lower: float = 99,
                 quality_upper: float = 100,
                 compression_type: ImgCmpAlb.ImageCompressionType = ImgCmpAlb.ImageCompressionType.JPEG):
        super().__init__(
            ImgCmpAlb(quality_lower=quality_lower,
                      quality_upper=quality_upper,
                      compression_type=compression_type,
                      always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
