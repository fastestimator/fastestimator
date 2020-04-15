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
import os
from typing import Any, Dict, Iterable, List, Union

from scipy.io import loadmat

from fastestimator.op.numpyop.numpyop import NumpyOp


class ReadMat(NumpyOp):
    """A class for reading .mat files from disk.

    This expects every sample to have a separate .mat file.

    Args:
        file: Dictionary key that contains the .mat path.
        keys: Key(s) to read from the .mat file.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        parent_path: Parent path that will be prepended to a given filepath.
    """
    def __init__(self,
                 file: str,
                 keys: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 parent_path: str = ""):
        super().__init__(inputs=file, outputs=keys, mode=mode)
        self.parent_path = parent_path
        self.out_list = True

    def forward(self, data: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = loadmat(os.path.normpath(os.path.join(self.parent_path, data)))
        results = [data[key] for key in self.outputs]
        return results
