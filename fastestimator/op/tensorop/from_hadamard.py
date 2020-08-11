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
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
from scipy.linalg import hadamard

from fastestimator.backend.matmul import matmul
from fastestimator.backend.maximum import maximum
from fastestimator.backend.reduce_sum import reduce_sum
from fastestimator.backend.reshape import reshape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.transpose import transpose
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class FromHadamard(TensorOp):
    """Convert hadamard encoded class representations into onehot probabilities.

    Args:
        inputs: Key of the input tensor(s) to be converted.
        outputs: Key of the output tensor(s) as class probabilities.
        n_classes: How many classes are there in the inputs.
        code_length: What code length to use. Will default to the smallest power of 2 which is >= the number of classes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 n_classes: int,
                 code_length: Optional[int] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True
        self.n_classes = n_classes
        if code_length is None:
            code_length = 1 << (n_classes - 1).bit_length()
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}")
        self.code_length = code_length
        self.labels = None
        self.baseline = None

    def build(self, framework: str) -> None:
        labels = hadamard(self.code_length).astype(np.float32)
        labels[np.arange(0, self.code_length, 2), 0] = -1  # Make first column alternate
        labels = labels[:self.n_classes]
        self.labels = transpose(to_tensor(labels, target_type=framework))
        self.baseline = to_tensor(np.array(0.0).astype('float32'), target_type=framework)
        if framework == "torch":
            self.labels = self.labels.to("cuda:0" if torch.cuda.is_available() else "cpu")
            self.baseline = self.baseline.to("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        results = []
        for elem in data:
            x = matmul(elem, self.labels)
            x = maximum(x, self.baseline) + 1e-6
            x = x / reshape(reduce_sum(x, axis=1), shape=[-1, 1])
            results.append(x)
        return results
