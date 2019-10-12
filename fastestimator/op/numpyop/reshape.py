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
import numpy as np

from fastestimator.op import NumpyOp


class Reshape(NumpyOp):
    """Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self, shape, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shape = shape

    def forward(self, data, state):
        """Reshapes data array

        Args:
            data: Data to be reshaped
            state: A dictionary containing background information such as 'mode'

        Returns:
            Reshaped array
        """
        data = np.reshape(data, self.shape)
        return data
