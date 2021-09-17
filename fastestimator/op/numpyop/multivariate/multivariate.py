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
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from albumentations import BboxParams, Compose, DualTransform, KeypointParams

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class MultiVariateAlbumentation(NumpyOp):
    """A base class for the DualTransform albumentation functions.

     DualTransforms are functions which apply simultaneously to images and corresponding information such as masks
     and/or bounding boxes.

    This is a wrapper for functionality provided by the Albumentations library:
    https://github.com/albumentations-team/albumentations. A useful visualization tool for many of the possible effects
    it provides is available at https://albumentations-demo.herokuapp.com.

    Args:
        func: An Albumentation function to be invoked.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The key of masks to be modified (with the same random factors as the image).
        bbox_in: The key of a bounding box(es) to be modified (with the same random factors as the image).
        keypoints_in: The key of keypoints to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The key to write the modified masks (defaults to `masks_in` if None).
        bbox_out: The key to write the modified bounding box(es) (defaults to `bbox_in` if None).
        keypoints_out: The key to write the modified keypoints (defaults to `keypoints_in` if None).
        bbox_params: Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').
        keypoint_params: Parameters defining the type of keypoints ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa').

    Raises:
        AssertionError: If none of the various inputs such as `image_in` or `mask_in` are provided.
    """
    def __init__(self,
                 func: DualTransform,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
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
                 keypoint_params: Union[KeypointParams, str, None] = None,
                 extra_in_keys: Optional[Dict[str, str]] = None,
                 extra_out_keys: Optional[Dict[str, str]] = None):
        assert any((image_in, mask_in, masks_in, bbox_in, keypoints_in)), "At least one input must be non-None"
        image_out = image_out or image_in
        mask_out = mask_out or mask_in
        masks_out = masks_out or masks_in
        bbox_out = bbox_out or bbox_in
        keypoints_out = keypoints_out or keypoints_in
        keys = OrderedDict([("image", image_in), ("mask", mask_in), ("masks", masks_in), ("bboxes", bbox_in),
                            ("keypoints", keypoints_in)])
        if extra_in_keys:
            keys.update(extra_in_keys)
        self.keys_in = OrderedDict([(k, v) for k, v in keys.items() if v is not None])
        keys = OrderedDict([("image", image_out), ("mask", mask_out), ("masks", masks_out), ("bboxes", bbox_out),
                            ("keypoints", keypoints_out)])
        if extra_out_keys:
            keys.update(extra_out_keys)
        self.keys_out = OrderedDict([(k, v) for k, v in keys.items() if v is not None])
        super().__init__(inputs=list(self.keys_in.values()),
                         outputs=list(self.keys_out.values()),
                         mode=mode,
                         ds_id=ds_id)
        if isinstance(bbox_params, str):
            bbox_params = BboxParams(bbox_params)
        if isinstance(keypoint_params, str):
            keypoint_params = KeypointParams(keypoint_params)
        self.func = Compose(transforms=[func], bbox_params=bbox_params, keypoint_params=keypoint_params)

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        result = self.func(**{k: v for k, v in zip(self.keys_in.keys(), data)})
        return [result[k] for k in self.keys_out.keys()]
