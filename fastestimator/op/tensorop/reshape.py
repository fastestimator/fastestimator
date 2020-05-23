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
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class Reshape(TensorOp):
    """Reshape a input tensor to conform to a given shape.

    Args:
        inputs: Key of the input tensor that is to be reshaped.
        outputs: Key of the output tensor that has been reshaped.
        shape: Target shape.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 shape: Union[int, Tuple[int, ...]],
                 mode: Union[None, str, Iterable[str]] = "!infer") -> None:

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shape = shape
        self.in_list, self.out_list = False, False

    def forward(self, data: Tensor, state: Dict[str, Any]) -> Tensor:
        if isinstance(data, tf.Tensor):
            return tf.reshape(data, self.shape)
        elif isinstance(data, torch.Tensor):
            return data.view(self.shape)
        else:
            raise ValueError("unrecognized data format: {}".format(type(data)))
