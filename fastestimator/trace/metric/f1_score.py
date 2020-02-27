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

from typing import List, TypeVar, Optional, Union

import numpy as np
from sklearn.metrics import f1_score

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class F1Score(Trace):
    """Calculate F1 score for classification task and report it back to logger.
    Args:
        true_key: Name of the keys in the ground truth label in data pipeline.
        pred_key: Name of the keys in predicted label. Default is `None`.
        labels: The set of labels to include. For more details, please refer to
            sklearn.netrics.f1_score. Defaults to None.
        pos_label: The class to report. For more details, please refer to
            sklearn.netrics.f1_score. Defaults to 1.
        average: It should be one of {"auto", "binary", "micro", "macro", "weighted", "samples", None}.
            If "auto", the trace will detect the input data type and choose the right average for you. Otherwise, it
            will pass its to sklearn.metrics.f1_score. Defaults to "auto".
        sample_weight: Sample weights, For more details, please refer to
            sklearn.netrics.f1_score. Defaults to None.
        mode: Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
        output_name: Name of the key to store back to the state. Defaults to "f1score".
    """
    def __init__(self,
                 true_key: str,
                 pred_key: Optional[str] = None,
                 labels: Optional[str] = None,
                 pos_label: Optional[Union[int, str]] = 1,
                 average: Optional[str] = 'auto',
                 sample_weight: Optional[np.ndarray] = None,
                 mode: Optional[str] = "eval",
                 output_name: Optional[str] = "f1score"):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self.output_name = output_name

    @property
    def true_key(self):
        return self.inputs[0]

    @property
    def pred_key(self):
        return self.inputs[1]

    def on_epoch_begin(self, data: Data):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data: Data):
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if y_true.shape[-1] > 1 and len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=-1)
        binary_classification = y_pred.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(y_pred)
        else:
            prediction_label = np.argmax(y_pred, axis=-1)
        self.binary_classification = binary_classification or y_pred.shape[-1] == 2
        assert prediction_label.size == y_true.size
        self.y_pred.extend(list(prediction_label.ravel()))
        self.y_true.extend(list(y_true.ravel()))

    def on_epoch_end(self, data: Data):
        if self.average == 'auto':
            if self.binary_classification:
                score = f1_score(np.ravel(self.y_true),
                                 np.ravel(self.y_pred),
                                 self.labels,
                                 self.pos_label,
                                 average='binary',
                                 sample_weight=self.sample_weight)
            else:
                score = f1_score(np.ravel(self.y_true),
                                 np.ravel(self.y_pred),
                                 self.labels,
                                 self.pos_label,
                                 average=None,
                                 sample_weight=self.sample_weight)
        else:
            score = f1_score(np.ravel(self.y_true),
                             np.ravel(self.y_pred),
                             self.labels,
                             self.pos_label,
                             self.average,
                             self.sample_weight)
        data[self.output_name] = score