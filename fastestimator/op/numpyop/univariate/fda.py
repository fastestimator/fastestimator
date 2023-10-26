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
from typing import Iterable, Tuple, Union, Any, Callable

from albumentations.augmentations import FDA as FDAAlb

from fastestimator.util.traceability_util import traceable

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation



@traceable()
class FDA(ImageOnlyAlbumentation):
    """Fourier Domain Adaptation. A simple style transfer can be implemented using this op.

    Args:
        reference_images: Sequence of objects that will be converted to images by read_fn. Can either be path
            of images or numpy arrays (depends upon read_fn).
        beta_limit: Beta coefficient for FDA.
        read_fn: User-defined function to read image, tensor or numpy array. Function should get an element
            of reference_images
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
                 beta_limit: Union[float, Tuple[float, float]] = 0.1,
                 read_fn: Callable = lambda x: x, # for reading tensor to numpy array
                 ):
        super().__init__(FDAAlb(reference_images=reference_images,
                                beta_limit=beta_limit,
                                read_fn=read_fn,
                                always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)
