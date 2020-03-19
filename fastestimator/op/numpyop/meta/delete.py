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
from typing import Any, Dict, List, NoReturn, Union

import numpy as np

from fastestimator.op import NumpyOp


class Delete(NumpyOp):
    """Delete the key, value pairs in data dict.

        Args:
            keys: Existing keys to be deleted in data dict.
    """
    def __init__(self, keys: Union[str, List[str]]):
        super().__init__(inputs=keys)
        #self.out_list = True

    def forward(self, data: Union[np.ndarray, List[np.ndarray]], state: Dict[str, Any]) -> List:
        pass
        #return [None] * len(self.outputs)
