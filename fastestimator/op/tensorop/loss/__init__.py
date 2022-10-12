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
                                            submod_attrs={'cross_entropy': ['CrossEntropy'],
                                                          'dice_loss': ['DiceLoss'],
                                                          'hinge': ['Hinge'],
                                                          'focal_loss': ['FocalLoss'],
                                                          'l2_regularization': ['L2Regularizaton'],
                                                          'loss': ['LossOp'],
                                                          'mean_squared_error': ['MeanSquaredError'],
                                                          'super_loss': ['SuperLoss'],
                                                          'l1_loss': ['L1_Loss']
                                                          })

if TYPE_CHECKING:
    from fastestimator.op.tensorop.loss.cross_entropy import CrossEntropy
    from fastestimator.op.tensorop.loss.dice_loss import DiceLoss
    from fastestimator.op.tensorop.loss.focal_loss import FocalLoss
    from fastestimator.op.tensorop.loss.hinge import Hinge
    from fastestimator.op.tensorop.loss.l1_loss import L1_Loss
    from fastestimator.op.tensorop.loss.l2_regularization import L2Regularizaton
    from fastestimator.op.tensorop.loss.loss import LossOp
    from fastestimator.op.tensorop.loss.mean_squared_error import MeanSquaredError
    from fastestimator.op.tensorop.loss.super_loss import SuperLoss
