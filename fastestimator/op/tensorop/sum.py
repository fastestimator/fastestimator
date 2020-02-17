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
from fastestimator.util.util import to_list

from fastestimator.op import TensorOp


class Sum(TensorOp):
    def forward(self, data, state):
        """ This class is to be used to compute the sum of input data.

        Args:
            data: input data to be summed
            state:  Information about the current execution context.
        Returns:
            Summed input data
        """
        return tf.reduce_sum(to_list(data), axis=0)
