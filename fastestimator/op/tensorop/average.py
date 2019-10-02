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

from fastestimator.op import TensorOp


class Average(TensorOp):
    def forward(self, data, state):
        """ This class is to be used to compute the average of input data.

        Args:
            data: input data to be averaged
            state:  Information about the current execution context.
        Returns:
            Averaged input data
        """
        iterdata = data if isinstance(data, list) else list(data) if isinstance(data, tuple) else [data]
        num_entries = len(iterdata)
        result = 0.0
        for elem in iterdata:
            result += elem
        return result / num_entries
