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
from fastestimator.op.tensorop.average import Average
from fastestimator.op.tensorop.binarize import Binarize
from fastestimator.op.tensorop.minmax import Minmax
from fastestimator.op.tensorop.onehot import Onehot
from fastestimator.op.tensorop.reshape import Reshape
from fastestimator.op.tensorop.resize import Resize
from fastestimator.op.tensorop.scale import Scale
from fastestimator.op.tensorop.z_score import Zscore
from fastestimator.op.tensorop.loss import Loss, BinaryCrossentropy, MeanSquaredError, MixUpLoss, \
    SparseCategoricalCrossentropy
from fastestimator.op.tensorop.augmentation import Augmentation2D, AdversarialSample, MixUpBatch, CutMixBatch
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop.filter import TensorFilter, ScalarFilter
