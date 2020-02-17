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
import tensorflow as tf

from fastestimator.op import TensorOp


class Probability(TensorOp):
    """Operate a TensorOp with certain probability

    Args:
        tensor_op: TensorOp instance
        prob: float number which indicates the probability of execution
    """
    def __init__(self, tensor_op, prob=0.5):
        super().__init__(inputs=tensor_op.inputs, outputs=tensor_op.outputs, mode=tensor_op.mode)
        self.tensor_op = tensor_op
        self.prob = prob

    def forward(self, data, state):
        """Execute the operator with probability

        Args:
            data: Tensor to be resized.
            state: Information about the current execution context.

        Returns:
            output tensor.
        """
        if self.prob > tf.random.uniform([]):
            data = self.tensor_op.forward(data, state)
        return data
