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
# see a complete list of effects in albumentation page: https://albumentations-demo.herokuapp.com
from fastestimator.op.numpyop.multivariate.affine import Affine
from fastestimator.op.numpyop.multivariate.center_crop import CenterCrop
from fastestimator.op.numpyop.multivariate.crop import Crop
from fastestimator.op.numpyop.multivariate.crop_non_empty_mask_if_exists import CropNonEmptyMaskIfExists
from fastestimator.op.numpyop.multivariate.elastic_transform import ElasticTransform
from fastestimator.op.numpyop.multivariate.flip import Flip
from fastestimator.op.numpyop.multivariate.grid_distortion import GridDistortion
from fastestimator.op.numpyop.multivariate.horizontal_flip import HorizontalFlip
from fastestimator.op.numpyop.multivariate.longest_max_size import LongestMaxSize
from fastestimator.op.numpyop.multivariate.mask_dropout import MaskDropout
from fastestimator.op.numpyop.multivariate.optical_distortion import OpticalDistortion
from fastestimator.op.numpyop.multivariate.pad_if_needed import PadIfNeeded
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
