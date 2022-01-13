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
import torch
import numpy as np
import tensorflow as tf

from typing import Any, List, Dict, Tuple, TypeVar, Union, Optional

from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable
from fastestimator.backend.tensor_normalize import normalize

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class Normalize(TensorOp):
    """Normalize a input tensor.

    Args:
        inputs: Key of the input tensor that is to be normalized.
        outputs: Key of the output tensor that has been normalized.
        mean: The mean which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))
        std: The standard deviation which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))
        epsilon: Epsilon value which needs to be added to standard deviation to avoid divide zero.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, List[str]]=None,
                 outputs: Union[str, List[str]]=None,
                 mean: Union[None, float, Tuple[float, ...]] = None,
                 std: Union[None, float, Tuple[float, ...]] = None,
                 epsilon: float = 1e-7,
                 mode = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.epsilon = epsilon
        self.mean = mean
        self.std = std

        if (isinstance(self.mean, Tuple) and  isinstance(self.std, float)) or (isinstance(self.std, Tuple) and  isinstance(self.mean, float)) or (isinstance(self.mean, Tuple) and isinstance(self.std, Tuple) and len(self.mean)!=len(self.std)):
            raise ValueError("Both mean and std should have same dimensions.")

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.epsilon = self.epsilon.to(device)

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        return normalize(data, self.mean, self.std, self.epsilon)