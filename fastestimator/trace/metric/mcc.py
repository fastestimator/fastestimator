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

from typing import Optional

import numpy as np
from sklearn.metrics import matthews_corrcoef

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class MCC(Trace):
    """ A trace which computes the Matthews Correlation Coefficient for a given set of predictions. This is a preferable
        metric to accuracy or F1 score since it automatically corrects for class imbalances and does not depend on the
        choice of target class (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/). Ideal value is 1, a value of 0
        means your predictions are completely uncorrelated with the true data. A value less than zero implies
        anti-correlation (you should invert your classifier predictions in order to do better)

    Args:
        true_key: The key of the true (known) class values
        pred_key: The key of the predicted class values
        mode: What mode to execute in (None to execute in all modes)
        output_name: What to call the output from this trace (for example in the logger output)
    """
    def __init__(self, true_key: str, pred_key: str, mode: Optional[str] = "eval", output_name: str = "mcc"):
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name)
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
        if y_pred.shape[-1] == 1:
            label_pred = np.round(y_pred)
        else:
            label_pred = np.argmax(y_pred, axis=-1)
        assert label_pred.size == y_true.size
        self.y_true.extend(y_true)
        self.y_pred.extend(label_pred)

    def on_epoch_end(self, data: Data):
        data.write_with_log(self.outputs[0], matthews_corrcoef(y_true=self.y_true, y_pred=self.y_pred))
