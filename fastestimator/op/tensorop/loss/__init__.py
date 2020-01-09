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
from fastestimator.op.tensorop.loss.loss import Loss
from fastestimator.op.tensorop.loss.binary_crossentropy import BinaryCrossentropy
from fastestimator.op.tensorop.loss.mean_squared_error import MeanSquaredError
from fastestimator.op.tensorop.loss.mixup_loss import MixUpLoss
from fastestimator.op.tensorop.loss.sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from fastestimator.op.tensorop.loss.smooth_l1_loss import SmoothL1Loss
from fastestimator.op.tensorop.loss.weighted_dice_loss import WeightedDiceLoss