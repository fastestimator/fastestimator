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
from fastestimator.util.util import to_list


class TypeConverter(NumpyOp):
    """convert features to different data types

    Args:
        target_type (str): the target data type, the following types are available:
            * 'bool'
            * 'int8'
            * 'int16'
            * 'int32'
            * 'int64'
            * 'uint8'
            * 'uint16'
            * 'uint32'
            * 'uint64'
            * 'float16'
            * 'float32'
            * 'float64'
            * 'str'
    """
    def __init__(self, target_type, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        converter_fn_map = {
            "bool": np.bool,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "str": np.str
        }
        assert target_type in converter_fn_map, "unsupported dtype for '{}'".format(target_type)
        self.convert_fn = converter_fn_map[target_type]

    def forward(self, data, state):
        data = to_list(data)
        for idx, elem in enumerate(data):
            data[idx] = self.convert_fn(elem)
        return data
