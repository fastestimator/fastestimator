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
from fastestimator.op.numpyop.univariate.binarize import Binarize
from fastestimator.op.numpyop.univariate.blur import Blur
from fastestimator.op.numpyop.univariate.channel_dropout import ChannelDropout
from fastestimator.op.numpyop.univariate.channel_shuffle import ChannelShuffle
from fastestimator.op.numpyop.univariate.channel_transpose import ChannelTranspose
from fastestimator.op.numpyop.univariate.clahe import CLAHE
from fastestimator.op.numpyop.univariate.coarse_dropout import CoarseDropout
from fastestimator.op.numpyop.univariate.downscale import Downscale
from fastestimator.op.numpyop.univariate.equalize import Equalize
from fastestimator.op.numpyop.univariate.expand_dims import ExpandDims
from fastestimator.op.numpyop.univariate.from_float import FromFloat
from fastestimator.op.numpyop.univariate.gaussian_blur import GaussianBlur
from fastestimator.op.numpyop.univariate.gaussian_noise import GaussianNoise
from fastestimator.op.numpyop.univariate.hue_saturation_value import HueSaturationValue
from fastestimator.op.numpyop.univariate.image_compression import ImageCompression
from fastestimator.op.numpyop.univariate.invert_img import InvertImg
from fastestimator.op.numpyop.univariate.iso_noise import ISONoise
from fastestimator.op.numpyop.univariate.median_blur import MedianBlur
from fastestimator.op.numpyop.univariate.minmax import Minmax
from fastestimator.op.numpyop.univariate.motion_blur import MotionBlur
from fastestimator.op.numpyop.univariate.multiplicative_noise import MultiplicativeNoise
from fastestimator.op.numpyop.univariate.normalize import Normalize
from fastestimator.op.numpyop.univariate.onehot import Onehot
from fastestimator.op.numpyop.univariate.pad_sequence import PadSequence
from fastestimator.op.numpyop.univariate.posterize import Posterize
from fastestimator.op.numpyop.univariate.random_brightness_contrast import RandomBrightnessContrast
from fastestimator.op.numpyop.univariate.random_fog import RandomFog
from fastestimator.op.numpyop.univariate.random_gamma import RandomGamma
from fastestimator.op.numpyop.univariate.random_rain import RandomRain
from fastestimator.op.numpyop.univariate.random_shadow import RandomShadow
from fastestimator.op.numpyop.univariate.random_snow import RandomSnow
from fastestimator.op.numpyop.univariate.random_sun_flare import RandomSunFlare
from fastestimator.op.numpyop.univariate.read_image import ReadImage
from fastestimator.op.numpyop.univariate.reshape import Reshape
from fastestimator.op.numpyop.univariate.rgb_shift import RGBShift
from fastestimator.op.numpyop.univariate.solarize import Solarize
from fastestimator.op.numpyop.univariate.to_array import ToArray
from fastestimator.op.numpyop.univariate.to_float import ToFloat
from fastestimator.op.numpyop.univariate.to_gray import ToGray
from fastestimator.op.numpyop.univariate.to_sepia import ToSepia
from fastestimator.op.numpyop.univariate.tokenize import Tokenize
from fastestimator.op.numpyop.univariate.word_to_id import WordtoId
