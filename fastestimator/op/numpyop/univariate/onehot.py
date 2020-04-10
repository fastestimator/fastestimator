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
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp


class Onehot(NumpyOp):
    """ Transform the label integer to one-hot-encoding
        Ref: https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

    Args:
        num_classes: Total number of classes.
        label_smoothing: Smoothing factor, after smoothing class output is: 1 - label_smoothing + label_smoothing
            / num_classes, the other class output is: label_smoothing / num_classes
        inputs: Input key(s) of labels to be onehot encoded
        outputs: Output key(s) of labels
        mode: What execution mode (train, eval, None) to apply this operation
    """
    def __init__(self,
                 num_classes: int,
                 label_smoothing: float = 0.0,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.in_list, self.out_list = True, True

    def forward(self, data: List[Union[int, np.ndarray]], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_onehot(elem) for elem in data]

    def _apply_onehot(self, data: Union[int, np.ndarray]) -> np.ndarray:
        class_index = np.array(data)
        assert "int" in str(class_index.dtype)
        assert class_index.size == 1, "data must have only one item"
        class_index = class_index.item()
        assert class_index < self.num_classes, "label value should be smaller than num_classes"
        output = np.full((self.num_classes), fill_value=self.label_smoothing / self.num_classes)
        output[class_index] = 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes
        return output
