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
from collections import OrderedDict
from typing import Union, Optional

from albumentations import Compose, BasicTransform

from fastestimator.op import NumpyOp


class ImageAugmentation(NumpyOp):
    """
    A class which wraps image augmentation methods provided by the albumentations library
    Args:
        albumentation (Compose, BasicTransform): The albumentation transform(s) to be applied to the data
        image (str): key corresponding to the image to be augmented
        mask (str): key corresponding to the mask to be augmented
        mask (str): key corresponding to the masks to be augmented
        bboxes (str): key corresponding to the bboxes to be augmented
        keypoints (str): key corresponding to the keypoints to be augmented
        mode (str): When to apply the augmentation proceedure (default "train")
    """
    def __init__(self,
                 albumentation: Union[Compose, BasicTransform],
                 image: Optional[str] = None,
                 mask: Optional[str] = None,
                 masks: Optional[str] = None,
                 bboxes: Optional[str] = None,
                 keypoints: Optional[str] = None,
                 mode: Optional[str] = "train"):
        self.albumentation = albumentation
        keys = OrderedDict([("image", image), ("mask", mask), ("masks", masks), ("bboxes", bboxes),
                            ("keypoints", keypoints)])
        self.keys = OrderedDict([(k, v) for k, v in keys.items() if v is not None])
        super().__init__(inputs=list(self.keys.values()), outputs=list(self.keys.values()), mode=mode)

    def forward(self, data, state):
        result = self.albumentation(**{k: v for k, v in zip(self.keys.keys(), data)})
        return [result[k] for k in self.keys.keys()]
