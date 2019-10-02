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

from scipy.io import loadmat

from fastestimator.op import NumpyOp


class MatReader(NumpyOp):
    """Class for reading .mat files.

    Args:
        parent_path: Parent path that will be added on given path.
    """
    def __init__(self, inputs=None, outputs=None, mode=None, parent_path=""):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self._loadmat = loadmat
        self.parent_path = parent_path

    def forward(self, data, state):
        """Reads mat file as dict.

        Args:
            data: Path to the mat file.
            state: A dictionary containing background information such as 'mode'

        Returns:
           dict
        """
        path = os.path.normpath(os.path.join(self.parent_path, data))
        data = self._loadmat(path)
        return data
