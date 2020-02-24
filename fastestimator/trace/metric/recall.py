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
from typing import List, Union

import numpy as np
from sklearn.metrics import recall_score

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class Recall(Trace):
    """Compute recall for classification task and report it back to logger.

    Args:
        true_key: Name of the keys in the ground truth label in data pipeline.
        pred_key: Name of the keys in predicted label. Defaults to None.
        mode: Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
        output_name: Name of the key to store to the state. Defaults to "recall".
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[str, List[str]] = ("eval", "test"),
                 output_name: str = "recall"):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.binary_classification = None
        self.y_true = []
        self.y_pred = []

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
        self.binary_classification = y_pred.shape[-1] == 1
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.round(y_pred)
        assert y_pred.size == y_true.size
        self.y_pred.extend(y_pred.ravel())
        self.y_true.extend(y_true.ravel())

    def on_epoch_end(self, data: Data):
        if self.binary_classification:
            score = recall_score(self.y_true, self.y_pred, average='binary')
        else:
            score = recall_score(self.y_true, self.y_pred, average=None)
        data.write_with_log(self.outputs[0], score)
