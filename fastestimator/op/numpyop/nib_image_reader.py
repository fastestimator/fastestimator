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

import nibabel as nib

from fastestimator.op import NumpyOp


class NIBImageReader(NumpyOp):
    """Class for reading Nifti images usinng nibabel
    Args:
        parent_path (str): Parent path that will be added on given path
    """
    def __init__(self, inputs=None, outputs=None, mode=None, parent_path=""):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.parent_path = parent_path

    def forward(self, path, state):
        """Reads numpy array from image path
        Args:
            path: path of the image
            state: A dictionary containing background information such as 'mode'
        Returns:
           Image as numpy array
        """
        path = os.path.normpath(os.path.join(self.parent_path, path))
        img_nifti = nib.load(path)
        affine = img_nifti.affine
        return img_nifti.get_data(), affine
