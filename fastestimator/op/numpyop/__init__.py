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
from fastestimator.op.numpyop.meta import Delete, OneOf, Sometimes
from fastestimator.op.numpyop.multivariate import Affine, CenterCrop, Crop, CropNonEmptyMaskIfExists, \
    ElasticTransform, Flip, GridDistortion, HorizontalFlip, LongestMaxSize, MaskDropout, OpticalDistortion, \
    PadIfNeeded, RandomCrop, RandomCropNearBBox, RandomGridShuffle, RandomResizedCrop, RandomRotate90, RandomScale, \
    RandomSizedBBoxSafeCrop, RandomSizedCrop, Resize, Rotate, ShiftScaleRotate, SmallestMaxSize, Transpose, \
    VerticalFlip
from fastestimator.op.numpyop.univariate import CLAHE, Blur, ChannelDropout, ChannelShuffle, ChannelTranspose, \
    CoarseDropout, Downscale, Equalize, ExpandDims, FromFloat, GaussianBlur, GaussianNoise, HueSaturationValue, \
    ImageCompression, InvertImg, ISONoise, MedianBlur, Minmax, MotionBlur, MultiplicativeNoise, Normalize, Onehot, \
    Posterize, RandomBrightnessContrast, RandomRain, ReadImage, Reshape, RGBShift, ToFloat
