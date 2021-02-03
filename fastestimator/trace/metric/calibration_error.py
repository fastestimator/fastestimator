#  Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

from typing import Optional, Set, Union

import calibration as cal
import numpy as np

from fastestimator.summary.summary import ValWithError
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.util import to_number


class CalibrationError(Trace):
    """A trace which computes the calibration error for a given set of predictions.

    Unlike many common calibration error estimation algorithms, this one has actual theoretical bounds on the quality
    of its output: https://arxiv.org/pdf/1909.10155v1.pdf.

    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary.
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        output_name: What to call the output from this trace (for example in the logger output).
        method: Either 'marginal' or 'top-label'. 'marginal' calibration averages the calibration error over each class,
            whereas 'top-label' computes the error based on only the most confident predictions.
        confidence_interval: The calibration error confidence interval to be reported (estimated empirically). Should be
            in the range (0, 100), or else None to omit this extra calculation.
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 output_name: str = "calibration_error",
                 method: str = "marginal",
                 confidence_interval: Optional[int] = None):
        self.y_true = []
        self.y_pred = []
        assert method in ('marginal', 'top-label'), \
            f"CalibrationError 'method' must be either 'marginal' or 'top-label', but got {method}."
        self.method = method
        if confidence_interval is not None:
            assert 0 < confidence_interval < 100, \
                f"CalibrationError 'confidence_interval' must be between 0 and 100, but got {confidence_interval}."
            confidence_interval = 1.0 - confidence_interval / 100.0
        self.confidence_interval = confidence_interval
        super().__init__(inputs=[true_key, pred_key], outputs=output_name, mode=mode)

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        assert y_pred.shape[0] == y_true.shape[0]
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def on_epoch_end(self, data: Data) -> None:
        self.y_true = np.squeeze(np.stack(self.y_true))
        self.y_pred = np.stack(self.y_pred)
        mid = round(cal.get_calibration_error(probs=self.y_pred, labels=self.y_true, mode=self.method), 4)
        low = None
        high = None
        if self.confidence_interval is not None:
            low, _, high = cal.get_calibration_error_uncertainties(probs=self.y_pred, labels=self.y_true,
                                                                   mode=self.method,
                                                                   alpha=self.confidence_interval)
            low = round(low, 4)
            high = round(high, 4)
        data.write_with_log(self.outputs[0], ValWithError(low, mid, high) if low is not None else mid)
