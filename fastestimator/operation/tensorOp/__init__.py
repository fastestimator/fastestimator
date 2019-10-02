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
from fastestimator.operation.tensorOp.average import Average
from fastestimator.operation.tensorOp.binarize import Binarize
from fastestimator.operation.tensorOp.minmax import Minmax
from fastestimator.operation.tensorOp.onehot import Onehot
from fastestimator.operation.tensorOp.reshape import Reshape
from fastestimator.operation.tensorOp.resize import Resize
from fastestimator.operation.tensorOp.scale import Scale
from fastestimator.operation.tensorOp.z_score import Zscore
from fastestimator.operation.tensorOp.loss import BinaryCrossentropy, MeanSquaredError, MixUpLoss, \
    SparseCategoricalCrossentropy
from fastestimator.operation.tensorOp.augmentation import Augmentation2D, AdversarialSample, MixUpBatch
from fastestimator.operation.tensorOp.model import ModelOp
from fastestimator.operation.tensorOp.filter import TensorFilter, ScalarFilter
