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
from fastestimator.backend.abs import abs
from fastestimator.backend.argmax import argmax
from fastestimator.backend.binary_crossentropy import binary_crossentropy
from fastestimator.backend.cast import cast
from fastestimator.backend.categorical_crossentropy import categorical_crossentropy
from fastestimator.backend.check_nan import check_nan
from fastestimator.backend.clip_by_value import clip_by_value
from fastestimator.backend.concat import concat
from fastestimator.backend.expand_dims import expand_dims
from fastestimator.backend.feed_forward import feed_forward
from fastestimator.backend.gather import gather
from fastestimator.backend.gather_from_batch import gather_from_batch
from fastestimator.backend.get_gradient import get_gradient
from fastestimator.backend.get_image_dims import get_image_dims
from fastestimator.backend.get_lr import get_lr
from fastestimator.backend.hinge import hinge
from fastestimator.backend.iwd import iwd
from fastestimator.backend.load_model import load_model
from fastestimator.backend.matmul import matmul
from fastestimator.backend.maximum import maximum
from fastestimator.backend.mean_squared_error import mean_squared_error
from fastestimator.backend.percentile import percentile
from fastestimator.backend.permute import permute
from fastestimator.backend.random_normal_like import random_normal_like
from fastestimator.backend.reduce_max import reduce_max
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_min import reduce_min
from fastestimator.backend.reduce_sum import reduce_sum
from fastestimator.backend.reshape import reshape
from fastestimator.backend.roll import roll
from fastestimator.backend.save_model import save_model
from fastestimator.backend.set_lr import set_lr
from fastestimator.backend.sign import sign
from fastestimator.backend.sparse_categorical_crossentropy import sparse_categorical_crossentropy
from fastestimator.backend.squeeze import squeeze
from fastestimator.backend.tensor_pow import tensor_pow
from fastestimator.backend.tensor_round import tensor_round
from fastestimator.backend.tensor_sqrt import tensor_sqrt
from fastestimator.backend.to_shape import to_shape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.to_type import to_type
from fastestimator.backend.transpose import transpose
from fastestimator.backend.update_model import update_model
from fastestimator.backend.watch import watch
from fastestimator.backend.zeros_like import zeros_like
