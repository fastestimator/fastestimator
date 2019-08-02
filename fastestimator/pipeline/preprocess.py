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
import tensorflow as tf
from fastestimator.util.op import TensorOp

epsilon = 1e-7

class Binarize(TensorOp):
    """
    Binarize data based on threshold between 0 and 1

    Args:
        threshold: Threshold for binarizing
    """
    def __init__(self, threshold):
        self.thresh = threshold

    def forward(self, data, ):
        """
        Transforms the image to binary based on threshold

        Args:
            data: Data to be binarized

        Returns:
            Tensor containing binarized data
        """
        data = tf.math.greater(data, self.thresh)
        data = tf.cast(data, tf.float32)
        return data


class Zscore(TensorOp):
    """
    Standardize data using zscore method
    """
    def forward(self, data):
        """
        Standardizes the data tensor

        Args:
            data: Data to be standardized

        Returns:
            Tensor containing standardized data
        """
        data = tf.cast(data, tf.float32)
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(
            tf.subtract(data, mean),
            tf.maximum(std, epsilon))
        data = tf.cast(data, tf.float32)
        return data


class Minmax(TensorOp):
    """
    Normalize data using the minmax method
    """
    def forward(self, data):
        """
        Normalizes the data tensor

        Args:
            data: Data to be normalized

        Returns:
            Tensor after minmax
        """
        data = tf.cast(data, tf.float32)
        data = tf.math.divide(
            tf.subtract(data, tf.reduce_min(data)),
            tf.maximum(
                tf.subtract(tf.reduce_max(data), tf.reduce_min(data)), epsilon))
        return data


class Scale(TensorOp):
    """
    Preprocessing class for scaling dataset

    Args:
        scalar: Scalar for scaling the data
    """
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, data):
        """
        Scales the data tensor

        Args:
            data: Data to be scaled

        Returns:
            Scaled data tensor
        """
        data = tf.cast(data, tf.float32)
        data = tf.scalar_mul(self.scalar, data)
        return data


class Onehot(TensorOp):
    """
    Preprocessing class for converting categorical labels to onehot encoding

    Args:
        num_dim: Number of dimensions of the labels
    """
    def __init__(self, num_dim, inputs=None, outputs=None, mode=None):
        self.num_dim = num_dim
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        """
        Transforms categorical labels to onehot encodings

        Args:
            data: Data to be preprocessed

        Returns:
            Transformed labels
        """
        data = tf.cast(data, tf.int32)
        data = tf.one_hot(data, self.num_dim)
        return data

    
class Resize(TensorOp):
    """
    Preprocessing class for resizing the images

    Args:
        size: Destination shape of the images
        resize_method: One of resize methods provided by tensorflow to be used
    """
    def __init__(self, size, resize_method=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.resize_method = resize_method

    def forward(self, data):
        """
        Resizes data tensor

        Args:
            data: Tensor to be resized

        Returns:
            Resized tensor
        """
        preprocessed_data = tf.image.resize_images(data, self.size, self.resize_method)
        return preprocessed_data


class Reshape(TensorOp):
    """
    Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self, shape):
        self.shape = shape

    def forward(self, data):
        """
        Reshapes data tensor
        
        Args:
            data: Data to be reshaped

        Returns:
            Reshaped tensor
        """
        data = tf.reshape(data, self.shape)
        return data
