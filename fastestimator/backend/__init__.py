# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submod_attrs={'_abs': ['abs'],
                                                          '_argmax': ['argmax'],
                                                          '_binary_crossentropy': ['binary_crossentropy'],
                                                          '_cast': ['cast'],
                                                          '_categorical_crossentropy': ['categorical_crossentropy'],
                                                          '_check_nan': ['check_nan'],
                                                          '_clip_by_value': ['clip_by_value'],
                                                          '_concat': ['concat'],
                                                          '_convert_tensor_precision': ['convert_tensor_precision'],
                                                          '_dice_score': ['dice_score'],
                                                          '_exp': ['exp'],
                                                          '_expand_dims': ['expand_dims'],
                                                          '_feed_forward': ['feed_forward'],
                                                          '_flip': ['flip'],
                                                          '_focal_loss': ['focal_loss'],
                                                          '_gather': ['gather'],
                                                          '_gather_from_batch': ['gather_from_batch'],
                                                          '_get_gradient': ['get_gradient'],
                                                          '_get_image_dims': ['get_image_dims'],
                                                          '_get_lr': ['get_lr'],
                                                          '_get_shape': ['get_shape'],
                                                          '_hinge': ['hinge'],
                                                          '_huber': ['huber'],
                                                          '_iwd': ['iwd'],
                                                          '_l1_loss': ['l1_loss'],
                                                          '_lambertw': ['lambertw'],
                                                          '_load_model': ['load_model'],
                                                          '_matmul': ['matmul'],
                                                          '_maximum': ['maximum'],
                                                          '_mean_squared_error': ['mean_squared_error'],
                                                          '_ones_like': ['ones_like'],
                                                          '_percentile': ['percentile'],
                                                          '_permute': ['permute'],
                                                          '_pow': ['pow'],
                                                          '_random_normal_like': ['random_normal_like'],
                                                          '_random_uniform_like': ['random_uniform_like'],
                                                          '_reduce_max': ['reduce_max'],
                                                          '_reduce_mean': ['reduce_mean'],
                                                          '_reduce_min': ['reduce_min'],
                                                          '_reduce_std': ['reduce_std'],
                                                          '_reduce_sum': ['reduce_sum'],
                                                          '_reshape': ['reshape'],
                                                          '_resize3d': ['resize_3d'],
                                                          '_roll': ['roll'],
                                                          '_save_model': ['save_model'],
                                                          '_set_lr': ['set_lr'],
                                                          '_sign': ['sign'],
                                                          '_smooth_l1_loss': ['smooth_l1_loss'],
                                                          '_sparse_categorical_crossentropy': [
                                                              'sparse_categorical_crossentropy'],
                                                          '_squeeze': ['squeeze'],
                                                          '_tensor_normalize': ['normalize'],
                                                          '_tensor_pow': ['tensor_pow'],
                                                          '_tensor_round': ['tensor_round'],
                                                          '_tensor_sqrt': ['tensor_sqrt'],
                                                          '_to_shape': ['to_shape'],
                                                          '_to_tensor': ['to_tensor'],
                                                          '_to_type': ['to_type'],
                                                          '_transpose': ['transpose'],
                                                          '_update_model': ['update_model'],
                                                          '_watch': ['watch'],
                                                          '_where': ['where'],
                                                          '_zeros_like': ['zeros_like'],
                                                          '_zscore': ['zscore'],
                                                          })

if TYPE_CHECKING:
    from fastestimator.backend._abs import abs
    from fastestimator.backend._argmax import argmax
    from fastestimator.backend._binary_crossentropy import binary_crossentropy
    from fastestimator.backend._cast import cast
    from fastestimator.backend._categorical_crossentropy import categorical_crossentropy
    from fastestimator.backend._check_nan import check_nan
    from fastestimator.backend._clip_by_value import clip_by_value
    from fastestimator.backend._concat import concat
    from fastestimator.backend._convert_tensor_precision import convert_tensor_precision
    from fastestimator.backend._dice_score import dice_score
    from fastestimator.backend._exp import exp
    from fastestimator.backend._expand_dims import expand_dims
    from fastestimator.backend._feed_forward import feed_forward
    from fastestimator.backend._flip import flip
    from fastestimator.backend._focal_loss import focal_loss
    from fastestimator.backend._gather import gather
    from fastestimator.backend._gather_from_batch import gather_from_batch
    from fastestimator.backend._get_gradient import get_gradient
    from fastestimator.backend._get_image_dims import get_image_dims
    from fastestimator.backend._get_lr import get_lr
    from fastestimator.backend._get_shape import get_shape
    from fastestimator.backend._hinge import hinge
    from fastestimator.backend._huber import huber
    from fastestimator.backend._iwd import iwd
    from fastestimator.backend._l1_loss import l1_loss
    from fastestimator.backend._l2_regularization import l2_regularization
    from fastestimator.backend._lambertw import lambertw
    from fastestimator.backend._load_model import load_model
    from fastestimator.backend._matmul import matmul
    from fastestimator.backend._maximum import maximum
    from fastestimator.backend._mean_squared_error import mean_squared_error
    from fastestimator.backend._ones_like import ones_like
    from fastestimator.backend._percentile import percentile
    from fastestimator.backend._permute import permute
    from fastestimator.backend._pow import pow
    from fastestimator.backend._random_normal_like import random_normal_like
    from fastestimator.backend._random_uniform_like import random_uniform_like
    from fastestimator.backend._reduce_max import reduce_max
    from fastestimator.backend._reduce_mean import reduce_mean
    from fastestimator.backend._reduce_min import reduce_min
    from fastestimator.backend._reduce_std import reduce_std
    from fastestimator.backend._reduce_sum import reduce_sum
    from fastestimator.backend._reshape import reshape
    from fastestimator.backend._resize3d import resize_3d
    from fastestimator.backend._roll import roll
    from fastestimator.backend._save_model import save_model
    from fastestimator.backend._set_lr import set_lr
    from fastestimator.backend._sign import sign
    from fastestimator.backend._smooth_l1_loss import smooth_l1_loss
    from fastestimator.backend._sparse_categorical_crossentropy import sparse_categorical_crossentropy
    from fastestimator.backend._squeeze import squeeze
    from fastestimator.backend._tensor_normalize import normalize
    from fastestimator.backend._tensor_pow import tensor_pow
    from fastestimator.backend._tensor_round import tensor_round
    from fastestimator.backend._tensor_sqrt import tensor_sqrt
    from fastestimator.backend._to_shape import to_shape
    from fastestimator.backend._to_tensor import to_tensor
    from fastestimator.backend._to_type import to_type
    from fastestimator.backend._transpose import transpose
    from fastestimator.backend._update_model import update_model
    from fastestimator.backend._watch import watch
    from fastestimator.backend._where import where
    from fastestimator.backend._zeros_like import zeros_like
    from fastestimator.backend._zscore import zscore
