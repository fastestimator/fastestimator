import os
import numpy as np

epsilon = 1e-7

class NumpyPreprocess:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        return data


class ImageReader(NumpyPreprocess):
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


class Zscore(NumpyPreprocess):
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
        data = (data - mean) / max(std, epsilon)
        return data


class Minmax(NumpyPreprocess):
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
        data = (data - data_min) / max((data_max - data_min), epsilon)
        return data


class Scale(NumpyPreprocess):
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


class Resize(NumpyPreprocess):
    def __init__(self, size, resize_method='bilinear', inputs=None, outputs=None, mode=None):
        import cv2
        self.size = (size[1], size[0])
        if resize_method == "bilinear":
            self.resize_method = cv2.INTER_LINEAR
        elif resize_method == "nearest":
            self.resize_method = cv2.INTER_NEAREST
        elif resize_method == "area":
            self.resize_method = cv2.INTER_AREA
        elif resize_method == "lanczos4":
            self.resize_method = cv2.INTER_LANCZOS4
        self.forward_fn = cv2.resize
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def transform(self, data):
        data = self.forward_fn(data, self.size, self.resize_method)
        return data


class Reshape(NumpyPreprocess):
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
