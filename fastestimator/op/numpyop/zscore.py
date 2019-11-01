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


class Zscore(NumpyOp):
    """Standardize data using zscore method
    """
    def __init__(self, inputs=None, outputs=None, mode=None, epsilon=1e-7):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.epsilon = epsilon

    def forward(self, data, state):
        """Standardizes the data

        Args:
            data: Data to be standardized
            state: A dictionary containing background information such as 'mode'

        Returns:
            Array containing standardized data
        """
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / max(std, self.epsilon)
        return data
