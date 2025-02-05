# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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
# see a complete list of effects in albumentation page: https://albumentations-demo.herokuapp.com
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submod_attrs={'affine': ['Affine'],
                                                          'bbox_safe_random_crop': ['BBoxSafeRandomCrop'],
                                                          'center_crop': ['CenterCrop'],
                                                          'crop_and_pad': ['CropAndPad'],
                                                          'crop': ['Crop'],
                                                          'crop_non_empty_mask_if_exists': ['CropNonEmptyMaskIfExists'],
                                                          'elastic_transform': ['ElasticTransform'],
                                                          'flip': ['Flip'],
                                                          'grid_distortion': ['GridDistortion'],
                                                          'horizontal_flip': ['HorizontalFlip'],
                                                          'longest_max_size': ['LongestMaxSize'],
                                                          'mask_dropout': ['MaskDropout'],
                                                          'xy_masking': ['XYMasking'],
                                                          'optical_distortion': ['OpticalDistortion'],
                                                          'pad_if_needed': ['PadIfNeeded'],
                                                          'random_crop_from_borders': ['RandomCropFromBorders'],
                                                          'random_crop': ['RandomCrop'],
                                                          'random_crop_near_bbox': ['RandomCropNearBBox'],
                                                          'random_grid_shuffle': ['RandomGridShuffle'],
                                                          'random_resized_crop': ['RandomResizedCrop'],
                                                          'random_rotate_90': ['RandomRotate90'],
                                                          'random_scale': ['RandomScale'],
                                                          'random_sized_bbox_safe_crop': ['RandomSizedBBoxSafeCrop'],
                                                          'random_sized_crop': ['RandomSizedCrop'],
                                                          'read_mat': ['ReadMat'],
                                                          'resize': ['Resize'],
                                                          'rotate': ['Rotate'],
                                                          'shift_scale_rotate': ['ShiftScaleRotate'],
                                                          'smallest_max_size': ['SmallestMaxSize'],
                                                          'transpose': ['Transpose'],
                                                          'vertical_flip': ['VerticalFlip'],
                                                          })

if TYPE_CHECKING:
    from fastestimator.op.numpyop.multivariate.affine import Affine
    from fastestimator.op.numpyop.multivariate.bbox_safe_random_crop import BBoxSafeRandomCrop
    from fastestimator.op.numpyop.multivariate.center_crop import CenterCrop
    from fastestimator.op.numpyop.multivariate.crop import Crop
    from fastestimator.op.numpyop.multivariate.crop_non_empty_mask_if_exists import CropNonEmptyMaskIfExists
    from fastestimator.op.numpyop.multivariate.elastic_transform import ElasticTransform
    from fastestimator.op.numpyop.multivariate.flip import Flip
    from fastestimator.op.numpyop.multivariate.grid_distortion import GridDistortion
    from fastestimator.op.numpyop.multivariate.horizontal_flip import HorizontalFlip
    from fastestimator.op.numpyop.multivariate.crop_and_pad import CropAndPad
    from fastestimator.op.numpyop.multivariate.longest_max_size import LongestMaxSize
    from fastestimator.op.numpyop.multivariate.mask_dropout import MaskDropout
    from fastestimator.op.numpyop.multivariate.xy_masking import XYMasking
    from fastestimator.op.numpyop.multivariate.optical_distortion import OpticalDistortion
    from fastestimator.op.numpyop.multivariate.pad_if_needed import PadIfNeeded
    from fastestimator.op.numpyop.multivariate.random_crop_from_borders import RandomCropFromBorders
    from fastestimator.op.numpyop.multivariate.random_crop import RandomCrop
    from fastestimator.op.numpyop.multivariate.random_crop_near_bbox import RandomCropNearBBox
    from fastestimator.op.numpyop.multivariate.random_grid_shuffle import RandomGridShuffle
    from fastestimator.op.numpyop.multivariate.random_resized_crop import RandomResizedCrop
    from fastestimator.op.numpyop.multivariate.random_rotate_90 import RandomRotate90
    from fastestimator.op.numpyop.multivariate.random_scale import RandomScale
    from fastestimator.op.numpyop.multivariate.random_sized_bbox_safe_crop import RandomSizedBBoxSafeCrop
    from fastestimator.op.numpyop.multivariate.random_sized_crop import RandomSizedCrop
    from fastestimator.op.numpyop.multivariate.read_mat import ReadMat
    from fastestimator.op.numpyop.multivariate.resize import Resize
    from fastestimator.op.numpyop.multivariate.rotate import Rotate
    from fastestimator.op.numpyop.multivariate.shift_scale_rotate import ShiftScaleRotate
    from fastestimator.op.numpyop.multivariate.smallest_max_size import SmallestMaxSize
    from fastestimator.op.numpyop.multivariate.transpose import Transpose
    from fastestimator.op.numpyop.multivariate.vertical_flip import VerticalFlip
