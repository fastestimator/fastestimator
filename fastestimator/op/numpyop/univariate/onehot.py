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
from typing import Any, Dict, Iterable, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class Onehot(NumpyOp):
    """Transform an integer label to one-hot-encoding.The label value start from 0.

    This can be desirable for increasing robustness against incorrect labels:
    https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

    Args:
        inputs: Input key(s) of labels to be onehot encoded.
        outputs: Output key(s) of labels.
        num_classes: Total number of classes.
        label_smoothing: Smoothing factor, after smoothing class output is: 1 - label_smoothing + label_smoothing
            / num_classes, the other class output is: label_smoothing / num_classes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 num_classes: int,
                 label_smoothing: float = 0.0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.in_list, self.out_list = True, True

    def forward(self, data: List[Union[int, np.ndarray]], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_onehot(elem) for elem in data]

    def _apply_onehot(self, data: Union[int, np.ndarray]) -> np.ndarray:
        data = np.atleast_1d(data)
        assert "int" in str(data.dtype).lower(), "data type must be an integer"

        max_class = np.max(data)
        assert max_class < self.num_classes, "label value should be smaller than num_classes"

        if data.size == 1:
            output = np.eye(self.num_classes, dtype=np.float32)[data[0]]
        else:
            output = np.eye(self.num_classes, dtype=np.float32)[data]

        if self.label_smoothing != 0:
            smooth_label = self.label_smoothing / self.num_classes
            output = np.where(output != 0, 1.0 - self.label_smoothing + smooth_label, smooth_label)

        return output
