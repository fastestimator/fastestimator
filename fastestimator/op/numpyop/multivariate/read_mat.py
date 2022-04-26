# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from fastestimator.util.traceability_util import traceable
from fastestimator.util.base_util import to_list


@traceable()
class ReadMat(NumpyOp):
    """A class for reading .mat files from disk.

    This expects every sample to have a separate .mat file.

    Args:
        inputs: Dictionary key that contains the .mat path.
        outputs: Keys to output from the mat file.
        mat_keys: (Optional) Keys to read from the .mat file. Defaults to `outputs`, but to re-name keys you can provide
            the original name here and the new name in `outputs`.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        parent_path: Parent path that will be prepended to a given filepath.
    """
    def __init__(self,
                 inputs: str,
                 outputs: Union[str, Iterable[str]],
                 mat_keys: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 parent_path: str = ""):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.parent_path = parent_path

        if mat_keys is None:
            self.mat_keys = self.outputs
        else:
            self.mat_keys = to_list(mat_keys)

        self.out_list = True

        if isinstance(self.mat_keys, List) and isinstance(self.outputs, List):
            assert len(self.mat_keys) == len(self.outputs), "keys and Output lengths must match"

    def forward(self, data: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_path = os.path.normpath(os.path.join(self.parent_path, data))
        data = loadmat(input_path)
        results = [data[key] for key in self.mat_keys]
        return results
