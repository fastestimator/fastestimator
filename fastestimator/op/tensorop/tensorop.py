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
from typing import Any, Dict, List, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.op.op import Op
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class TensorOp(Op):
    """An Operator class which takes and returns tensor data.

    These Operators are used in fe.Network to perform graph-based operations like neural network training.
    """
    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        """A method which will be invoked in order to transform data.

        This method will be invoked on batches of data.

        Args:
            data: The batch from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data
            dictionary based on whatever keys this Op declares as its `outputs`.
        """
        return data
