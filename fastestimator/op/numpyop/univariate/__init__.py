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
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submod_attrs={'autocontrast': ['AutoContrast'],
                                                          'binarize': ['Binarize'],
                                                          'blur': ['Blur'],
                                                          'brightness': ['Brightness'],
                                                          'calibate': ['Calibrate'],
                                                          'channel_dropout': ['ChannelDropout'],
                                                          'channel_shuffle': ['ChannelShuffle'],
                                                          'channel_transpose': ['ChannelTranspose'],
                                                          'clahe': ['CLAHE'],
                                                          'coarse_dropout': ['CoarseDropout'],
                                                          'color': ['Color'],
                                                          'color_jitter': ['ColorJitter'],
                                                          'contrast': ['Contrast'],
                                                          'downscale': ['Downscale'],
                                                          'equalize': ['Equalize'],
                                                          'expand_dims': ['ExpandDims'],
                                                          'from_float': ['FromFloat'],
                                                          'gaussian_blur': ['GaussianBlur'],
                                                          'gaussian_noise': ['GaussianNoise'],
                                                          'hadamard': ['Hadamard'],
                                                          'hue_saturation_value': ['HueSaturationValue'],
                                                          'iaa_additive_gaussian_noise': ['IAAAdditiveGaussianNoise'],
                                                          'image_compression': ['ImageCompression'],
                                                          'invert_img': ['InvertImg'],
                                                          'iso_noise': ['ISONoise'],
                                                          'median_blur': ['MedianBlur'],
                                                          'minmax': ['Minmax'],
                                                          'motion_blur': ['MotionBlur'],
                                                          'multiplicative_noise': ['MultiplicativeNoise'],
                                                          'normalize': ['Normalize'],
                                                          'onehot': ['Onehot'],
                                                          'pad_sequence': ['PadSequence'],
                                                          'posterize': ['Posterize'],
                                                          'random_brightness_contrast': ['RandomBrightnessContrast'],
                                                          'random_fog': ['RandomFog'],
                                                          'random_gamma': ['RandomGamma'],
                                                          'random_rain': ['RandomRain'],
                                                          'random_shadow': ['RandomShadow'],
                                                          'random_shapes': ['RandomShapes'],
                                                          'random_snow': ['RandomSnow'],
                                                          'random_sun_flare': ['RandomSunFlare'],
                                                          'read_image': ['ReadImage'],
                                                          'reshape': ['Reshape'],
                                                          'rgb_shift': ['RGBShift'],
                                                          'rua': ['RUA'],
                                                          'sharpness': ['Sharpness'],
                                                          'shear_x': ['ShearX'],
                                                          'shear_y': ['ShearY'],
                                                          'solarize': ['Solarize'],
                                                          'to_array': ['ToArray'],
                                                          'to_float': ['ToFloat'],
                                                          'to_gray': ['ToGray'],
                                                          'to_sepia': ['ToSepia'],
                                                          'tokenize': ['Tokenize'],
                                                          'translate_x': ['TranslateX'],
                                                          'translate_y': ['TranslateY'],
                                                          'word_to_id': ['WordtoId'],
                                                          })

if TYPE_CHECKING:
    from fastestimator.op.numpyop.univariate.autocontrast import AutoContrast
    from fastestimator.op.numpyop.univariate.binarize import Binarize
    from fastestimator.op.numpyop.univariate.blur import Blur
    from fastestimator.op.numpyop.univariate.brightness import Brightness
    from fastestimator.op.numpyop.univariate.calibate import Calibrate
    from fastestimator.op.numpyop.univariate.channel_dropout import ChannelDropout
    from fastestimator.op.numpyop.univariate.channel_shuffle import ChannelShuffle
    from fastestimator.op.numpyop.univariate.channel_transpose import ChannelTranspose
    from fastestimator.op.numpyop.univariate.clahe import CLAHE
    from fastestimator.op.numpyop.univariate.coarse_dropout import CoarseDropout
    from fastestimator.op.numpyop.univariate.color import Color
    from fastestimator.op.numpyop.univariate.color_jitter import ColorJitter
    from fastestimator.op.numpyop.univariate.contrast import Contrast
    from fastestimator.op.numpyop.univariate.downscale import Downscale
    from fastestimator.op.numpyop.univariate.equalize import Equalize
    from fastestimator.op.numpyop.univariate.expand_dims import ExpandDims
    from fastestimator.op.numpyop.univariate.from_float import FromFloat
    from fastestimator.op.numpyop.univariate.gaussian_blur import GaussianBlur
    from fastestimator.op.numpyop.univariate.gaussian_noise import GaussianNoise
    from fastestimator.op.numpyop.univariate.hadamard import Hadamard
    from fastestimator.op.numpyop.univariate.hue_saturation_value import HueSaturationValue
    from fastestimator.op.numpyop.univariate.iaa_additive_gaussian_noise import IAAAdditiveGaussianNoise
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
    from fastestimator.op.numpyop.univariate.random_shapes import RandomShapes
    from fastestimator.op.numpyop.univariate.random_snow import RandomSnow
    from fastestimator.op.numpyop.univariate.random_sun_flare import RandomSunFlare
    from fastestimator.op.numpyop.univariate.read_image import ReadImage
    from fastestimator.op.numpyop.univariate.reshape import Reshape
    from fastestimator.op.numpyop.univariate.rgb_shift import RGBShift
    from fastestimator.op.numpyop.univariate.rua import RUA
    from fastestimator.op.numpyop.univariate.sharpness import Sharpness
    from fastestimator.op.numpyop.univariate.shear_x import ShearX
    from fastestimator.op.numpyop.univariate.shear_y import ShearY
    from fastestimator.op.numpyop.univariate.solarize import Solarize
    from fastestimator.op.numpyop.univariate.to_array import ToArray
    from fastestimator.op.numpyop.univariate.to_float import ToFloat
    from fastestimator.op.numpyop.univariate.to_gray import ToGray
    from fastestimator.op.numpyop.univariate.to_sepia import ToSepia
    from fastestimator.op.numpyop.univariate.tokenize import Tokenize
    from fastestimator.op.numpyop.univariate.translate_x import TranslateX
    from fastestimator.op.numpyop.univariate.translate_y import TranslateY
    from fastestimator.op.numpyop.univariate.word_to_id import WordtoId
