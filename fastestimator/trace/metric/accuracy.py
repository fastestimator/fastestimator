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
from typing import Set, Union

import numpy as np

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data


class Accuracy(Trace):
    """A trace which computes the accuracy for a given set of predictions.

    Consider using MCC instead: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/

    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary.
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        output_name: What to call the output from this trace (for example in the logger output).
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 output_name: str = "accuracy") -> None:
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name)
        self.total = 0
        self.correct = 0

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.total = 0
        self.correct = 0

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.round(y_pred)
        assert y_pred.size == y_true.size
        self.correct += np.sum(y_pred.ravel() == y_true.ravel())
        self.total += len(y_pred.ravel())

    def on_epoch_end(self, data: Data) -> None:
        data.write_with_log(self.outputs[0], self.correct / self.total)
