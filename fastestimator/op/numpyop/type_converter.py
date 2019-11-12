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
from fastestimator.util.util import convert_np_dtype, to_list


class TypeConverter(NumpyOp):
    """convert features to different data types

    Args:
        target_type (str): the target data type, the following types are available:
    """
    def __init__(self, target_type, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.convert_fn = convert_np_dtype(target_type)

    def forward(self, data, state):
        data = to_list(data)
        for idx, elem in enumerate(data):
            data[idx] = self.convert_fn(elem)
        if len(data) == 1:
            data = data[0]
        return data
