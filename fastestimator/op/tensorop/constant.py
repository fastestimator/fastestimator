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
from fastestimator.util.util import to_list


class Constant(TensorOp):
    """ A class to introduce a constant into the state dictionary

    Args:
        shape_like (str): Key of a variable whose shape will be matched
        constant (int, float, func): A constant or function defining the value to be output. If a function is provided,
                                    the desired shape will be passed as an argument.
        outputs (str): The name of the output value
        mode (str): Which mode to run in ('train', 'eval', None)
    """
    def __init__(self, shape_like, constant=0, outputs=None, mode=None):
        super().__init__(inputs=shape_like, outputs=outputs, mode=mode)
        self.constant = constant

    def forward(self, data, state):
        """ This class is to be used to compute a constant of a particular shape

        Args:
            data: input data to define the constant's shape
            state: Information about the current execution context.
        Returns:
            a constant tensor of the same shape as the input
        """
        results = []
        for elem in to_list(data):
            if hasattr(self.constant, "__call__"):
                results.append(tf.zeros(elem.shape) + self.constant(elem.shape))
            else:
                results.append(tf.zeros(elem.shape) + self.constant)
        return results if len(results) > 1 else results[0]
