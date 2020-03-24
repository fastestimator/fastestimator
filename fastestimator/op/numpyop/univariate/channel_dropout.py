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

from albumentations.augmentations.transforms import ChannelDropout as ChannelDropoutAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class ChannelDropout(ImageOnlyAlbumentation):
    """Randomly drop channels from the image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            channel_drop_range: range from which we choose the number of channels to drop.
            fill_value: pixel value for the dropped channel.
        Image types:
            int8, uint16, unit32, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 channel_drop_range: Tuple[int, int] = (1, 1),
                 fill_value: Union[int, float] = 0):
        super().__init__(
            ChannelDropoutAlb(channel_drop_range=channel_drop_range, fill_value=fill_value, always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
