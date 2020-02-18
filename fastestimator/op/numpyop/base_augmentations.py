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
from copy import deepcopy
from typing import Optional, List, Union, Dict, Any

import numpy as np
from albumentations import ImageOnlyTransform, ReplayCompose, Compose, DualTransform, BboxParams, KeypointParams

from fastestimator.op import NumpyOp


class ImageOnlyAlbumentation(NumpyOp):
    def __init__(self,
                 func: ImageOnlyTransform,
                 inputs: Union[List[str], str, None] = None,
                 outputs: Union[List[str], str, None] = None,
                 mode: Optional[str] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(self.inputs, List) and isinstance(self.outputs, List):
            assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.func = Compose(transforms=[func])
        self.replay_func = ReplayCompose(transforms=[deepcopy(func)])

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        results = [self.replay_func(image=data[0])]
        for i in range(1, len(data)):
            results.append(self.replay_func.replay(results[0]['replay'], image=data[i]))
        return [result["image"] for result in results]


class MultiVariateAlbumentation(NumpyOp):
    """
    A base class for the DualTransform albumentation functions. Unfortunately these ops can't use the un-named
    input/output syntax since this would likely lead to problems when trying to match up outputs from one transform to
    the inputs from another (suppose the first works on image+mask and the second on image+bbox, or if no inputs were
    specified then how does the system differentiate a single image from a single mask, etc.)
    """
    def __init__(self,
                 func: DualTransform,
                 mode: Optional[str] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 bbox_in: Optional[str] = None,
                 keypoints_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None,
                 bbox_out: Optional[str] = None,
                 keypoints_out: Optional[str] = None,
                 bbox_params: Union[BboxParams, str, None] = None,
                 keypoint_params: Union[KeypointParams, str, None] = None):
        assert any((image_in, mask_in, masks_in, bbox_in, keypoints_in)), "At least one input must be non-None"
        image_out = image_out or image_in
        mask_out = mask_out or mask_in
        masks_out = masks_out or masks_in
        bbox_out = bbox_out or bbox_in
        keypoints_out = keypoints_out or keypoints_in
        keys = OrderedDict([("image", image_in), ("mask", mask_in), ("masks", masks_in), ("bboxes", bbox_in),
                            ("keypoints", keypoints_in)])
        self.keys_in = OrderedDict([(k, v) for k, v in keys.items() if v is not None])
        keys = OrderedDict([("image", image_out), ("mask", mask_out), ("masks", masks_out), ("bboxes", bbox_out),
                            ("keypoints", keypoints_out)])
        self.keys_out = OrderedDict([(k, v) for k, v in keys.items() if v is not None])
        super().__init__(inputs=list(self.keys_in.values()), outputs=list(self.keys_out.values()), mode=mode)
        if isinstance(bbox_params, str):
            bbox_params = BboxParams(bbox_params)
        if isinstance(keypoint_params, str):
            keypoint_params = KeypointParams(keypoint_params)
        self.func = Compose(transforms=[func], bbox_params=bbox_params, keypoint_params=keypoint_params)

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        result = self.func(**{k: v for k, v in zip(self.keys_in.keys(), data)})
        return [result[k] for k in self.keys_out.keys()]
