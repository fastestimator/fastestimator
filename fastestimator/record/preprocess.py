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
import numpy as np
from fastestimator.util.op import NumpyOp

EPSILON = 1e-7

class ImageReader(NumpyOp):
    """
    Class for reading png or jpg images

    Args:
        parent_path: Parent path that will be added on given path
        grey_scale: Boolean to indicate whether or not to read image as grayscale
    """
    def __init__(self, inputs=None, outputs=None, mode=None, parent_path="", grey_scale=False):
        import cv2
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.parent_path = parent_path
        self.color_flag = cv2.IMREAD_COLOR
        if grey_scale:
            self.color_flag = cv2.IMREAD_GRAYSCALE
        self.forward_fn = cv2.imread

    def forward(self, path):
        """
        Reads numpy array from image path

        Args:
            path: path of the image
            feature: Auxiliary data that may be used by the image reader

        Returns:
           Image as numpy array
        """
        path = os.path.normpath(os.path.join(self.parent_path, path))
        data = self.forward_fn(path, self.color_flag)
        if not isinstance(data, np.ndarray):
            raise ValueError('cv2 did not read correctly for file "{}"'.format(path))
        return data


class Zscore(NumpyOp):
    """
    Standardize data using zscore method
    """
    def forward(self, data):
        """
        Standardizes the data

        Args:
            data: Data to be standardized
            feature: Auxiliary data needed for the standardization

        Returns:
            Array containing standardized data
        """
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / max(std, EPSILON)
        return data


class Minmax(NumpyOp):
    """
    Normalize data using the minmax method
    """
    def forward(self, data):
        """
        Normalizes the data

        Args:
            data: Data to be normalized
            feature: Auxiliary data needed for the normalization

        Returns:
            Normalized numpy array
        """
        data_max = np.max(data)
        data_min = np.min(data)
        data = (data - data_min) / max((data_max - data_min), EPSILON)
        return data


class Scale(NumpyOp):
    """
    Preprocessing class for scaling dataset

    Args:
        scalar: Scalar for scaling the data
    """
    def __init__(self, scalar, inputs=None, outputs=None, mode=None):
        self.scalar = scalar
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        """
        Scales the data tensor

        Args:
            data: Data to be scaled
            feature: Auxiliary data needed for the normalization

        Returns:
            Scaled data array
        """
        data = self.scalar * data
        return data


class Reshape(NumpyOp):
    """
    Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self, shape, inputs=None, outputs=None, mode=None):
        self.shape = shape
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        """
        Reshapes data array

        Args:
            data: Data to be reshaped
            feature: Auxiliary data needed for the reshaping

        Returns:
            Reshaped array
        """
        data = np.reshape(data, self.shape)
        return data

class MatReader(NumpyOp):
    """Class for reading .mat files.

    Args:
        parent_path: Parent path that will be added on given path.
    """

    def __init__(self, inputs=None, outputs=None, mode=None, parent_path=""):

        from scipy.io import loadmat
        self._loadmat = loadmat
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.parent_path = parent_path

    def forward(self, data):
        """Reads mat file as dict.

        Args:
            data: Path to the mat file.
            feature: Auxiliary data that may be used.

        Returns:
           dict
        """
        path = os.path.normpath(os.path.join(self.parent_path, data))
        data = self._loadmat(data)
        return data

class Resize(NumpyOp):
    """Resize image.

    Args:
        target_size (tuple): Target image size in (height, width) format.
        resize_method (string): `bilinear`, `nearest`, `area`, and `lanczos4` are available.
        keep_ratio (bool): If `True`, the resulting image will be padded to keep the original aspect ratio.

    Returns:
        Resized `np.ndarray`.
    """

    def __init__(self, target_size, resize_method='bilinear', keep_ratio=False, inputs=None, outputs=None, mode=None):
        import cv2
        self._cv2 = cv2
        self.target_size = target_size
        if resize_method == "bilinear":
            self.resize_method = cv2.INTER_LINEAR
        elif resize_method == "nearest":
            self.resize_method = cv2.INTER_NEAREST
        elif resize_method == "area":
            self.resize_method = cv2.INTER_AREA
        elif resize_method == "lanczos4":
            self.resize_method = cv2.INTER_LANCZOS4
        self.keep_ratio = keep_ratio
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data, feature=None):
        if self.keep_ratio:
            original_ratio = data.shape[1] / data.shape[0]
            target_ratio = self.target_size[1] / self.target_size[0]
            if original_ratio >= target_ratio:
                pad = (data.shape[1] / target_ratio - data.shape[0]) / 2
                pad_boarder = (np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int), 0, 0)
            else:
                pad = (data.shape[0] * target_ratio - data.shape[1]) / 2
                pad_boarder = (0, 0, np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int))
            data = self._cv2.copyMakeBorder(data, *pad_boarder, self._cv2.BORDER_CONSTANT)
        data = self._cv2.resize(data, (self.target_size[1], self.target_size[0]), self.resize_method)
        return data