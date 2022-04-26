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
                                            submod_attrs={'accuracy': ['Accuracy'],
                                                          'bleu_score': ['BleuScore'],
                                                          'calibration_error': ['CalibrationError'],
                                                          'confusion_matrix': ['ConfusionMatrix'],
                                                          'dice': ['Dice'],
                                                          'f1_score': ['F1Score'],
                                                          'mcc': ['MCC'],
                                                          'mean_average_precision': ['MeanAveragePrecision'],
                                                          'precision': ['Precision'],
                                                          'recall': ['Recall'], })

if TYPE_CHECKING:
    from fastestimator.trace.metric.accuracy import Accuracy
    from fastestimator.trace.metric.bleu_score import BleuScore
    from fastestimator.trace.metric.calibration_error import CalibrationError
    from fastestimator.trace.metric.confusion_matrix import ConfusionMatrix
    from fastestimator.trace.metric.dice import Dice
    from fastestimator.trace.metric.f1_score import F1Score
    from fastestimator.trace.metric.mcc import MCC
    from fastestimator.trace.metric.mean_average_precision import MeanAveragePrecision
    from fastestimator.trace.metric.precision import Precision
    from fastestimator.trace.metric.recall import Recall
