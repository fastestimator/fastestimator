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
from fastestimator.op.numpyop.blur import Blur
from fastestimator.op.numpyop.center_crop import CenterCrop
from fastestimator.op.numpyop.channel_dropout import ChannelDropout
from fastestimator.op.numpyop.channel_shuffle import ChannelShuffle
from fastestimator.op.numpyop.channel_transpose import ChannelTranspose
from fastestimator.op.numpyop.clahe import CLAHE
from fastestimator.op.numpyop.coarse_dropout import CoarseDropout
from fastestimator.op.numpyop.crop import Crop
from fastestimator.op.numpyop.crop_non_empty_mask_if_exists import CropNonEmptyMaskIfExists
from fastestimator.op.numpyop.downscale import Downscale
from fastestimator.op.numpyop.elastic_transform import ElasticTransform
from fastestimator.op.numpyop.equalize import Equalize
from fastestimator.op.numpyop.expand_dims import ExpandDims
from fastestimator.op.numpyop.flip import Flip
from fastestimator.op.numpyop.from_float import FromFloat
from fastestimator.op.numpyop.gaussian_blur import GaussianBlur
from fastestimator.op.numpyop.gaussian_noise import GaussianNoise
from fastestimator.op.numpyop.grid_distortion import GridDistortion
from fastestimator.op.numpyop.horizontal_flip import HorizontalFlip
from fastestimator.op.numpyop.hue_saturation_value import HueSaturationValue
from fastestimator.op.numpyop.image_compression import ImageCompression
from fastestimator.op.numpyop.invert_img import InvertImg
from fastestimator.op.numpyop.iso_noise import ISONoise
from fastestimator.op.numpyop.longest_max_size import LongestMaxSize
from fastestimator.op.numpyop.mask_dropout import MaskDropout
from fastestimator.op.numpyop.median_blur import MedianBlur
from fastestimator.op.numpyop.minmax import Minmax
from fastestimator.op.numpyop.motion_blur import MotionBlur
from fastestimator.op.numpyop.multiplicative_noise import MultiplicativeNoise
from fastestimator.op.numpyop.normalize import Normalize
from fastestimator.op.numpyop.optical_distortion import OpticalDistortion
from fastestimator.op.numpyop.pad_if_needed import PadIfNeeded
from fastestimator.op.numpyop.posterize import Posterize
from fastestimator.op.numpyop.random_brightness_contrast import RandomBrightnessContrast
from fastestimator.op.numpyop.random_crop import RandomCrop
from fastestimator.op.numpyop.random_fog import RandomFog
from fastestimator.op.numpyop.random_gamma import RandomGamma
from fastestimator.op.numpyop.random_rain import RandomRain
from fastestimator.op.numpyop.random_rotate_90 import RandomRotate90
from fastestimator.op.numpyop.random_shadow import RandomShadow
from fastestimator.op.numpyop.random_snow import RandomSnow
from fastestimator.op.numpyop.random_sun_flare import RandomSunFlare
from fastestimator.op.numpyop.rgb_shift import RGBShift
from fastestimator.op.numpyop.solarize import Solarize
from fastestimator.op.numpyop.sometimes import Sometimes
from fastestimator.op.numpyop.to_float import ToFloat
from fastestimator.op.numpyop.to_gray import ToGray
from fastestimator.op.numpyop.to_sepia import ToSepia
from fastestimator.op.numpyop.vertical_flip import VerticalFlip
