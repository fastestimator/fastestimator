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

EPSILON = 1e-7


class TensorFilter(TensorOp):
    def __init__(self, inputs, mode="train"):
        super().__init__(inputs=inputs, mode=mode)

    def forward(self, data, state):
        return tf.constant(True)


class ScalarFilter(TensorFilter):
    """Class for performing filtering on dataset based on scalar values.

    Args:
        feature_name: Name of the key in the dataset that is to be filtered
        filter_value: The values in the dataset that are to be filtered.
        keep_prob: The probability of keeping the example
        mode: mode that the filter acts on
    """
    def __init__(self, inputs, filter_value, keep_prob, mode="train"):
        super().__init__(inputs=inputs, mode=mode)
        self.filter_value = filter_value
        self.keep_prob = keep_prob
        self._verify_inputs()

    def _verify_inputs(self):
        assert isinstance(self.inputs, str), "ScalarFilter only accepts single string input"
        self.filter_value = self._convert_to_list(self.filter_value)
        self.keep_prob = self._convert_to_list(self.keep_prob)
        assert len(self.filter_value) == len(self.keep_prob), "filter_value and keep_prob must be same length"

    def _convert_to_list(self, data):
        if not isinstance(data, list):
            if isinstance(data, tuple):
                data = list(data)
            else:
                data = [data]
        return data

    def forward(self, data, state):
        pass_filter = tf.constant(True)
        for filter_value, keep_prob in zip(self.filter_value, self.keep_prob):
            pass_current_filter = tf.cond(tf.equal(data, filter_value),
                                          lambda: tf.greater(keep_prob, tf.random.uniform([])),
                                          lambda: tf.constant(True))
            pass_filter = tf.logical_and(pass_filter, pass_current_filter)
        return pass_filter


class Binarize(TensorOp):
    """
    Binarize data based on threshold between 0 and 1

    Args:
        threshold: Threshold for binarizing
    """
    def __init__(self, threshold, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.thresh = threshold

    def forward(self, data, state):
        """
        Transforms the image to binary based on threshold

        Args:
            data: Data to be binarized
            state: Information about the current execution context

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
    def forward(self, data, state):
        """
        Standardizes the data tensor

        Args:
            data: Data to be standardized
            state: Information about the current execution context

        Returns:
            Tensor containing standardized data
        """
        data = tf.cast(data, tf.float32)
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(tf.subtract(data, mean), tf.maximum(std, EPSILON))
        data = tf.cast(data, tf.float32)
        return data


class Minmax(TensorOp):
    """
    Normalize data using the minmax method
    """
    def forward(self, data, state):
        """
        Normalizes the data tensor

        Args:
            data: Data to be normalized
            state: Information about the current execution context

        Returns:
            Tensor after minmax
        """
        data = tf.cast(data, tf.float32)
        data = tf.math.divide(tf.subtract(data, tf.reduce_min(data)),
                              tf.maximum(tf.subtract(tf.reduce_max(data), tf.reduce_min(data)), EPSILON))
        return data


class Scale(TensorOp):
    """
    Preprocessing class for scaling dataset

    Args:
        scalar: Scalar for scaling the data
    """
    def __init__(self, scalar, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.scalar = scalar

    def forward(self, data, state):
        """
        Scales the data tensor

        Args:
            data: Data to be scaled
            state: Information about the current execution context

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
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_dim = num_dim

    def forward(self, data, state):
        """
        Transforms categorical labels to onehot encodings

        Args:
            data: Data to be preprocessed
            state: Information about the current execution context

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
    def __init__(self, size, resize_method=tf.image.ResizeMethod.BILINEAR, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.size = size
        self.resize_method = resize_method

    def forward(self, data, state):
        """
        Resizes data tensor

        Args:
            data: Tensor to be resized
            state: Information about the current execution context

        Returns:
            Resized tensor
        """
        preprocessed_data = tf.image.resize(data, self.size, self.resize_method)
        return preprocessed_data


class Reshape(TensorOp):
    """
    Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self, shape, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shape = shape

    def forward(self, data, state):
        """
        Reshapes data tensor

        Args:
            data: Data to be reshaped
            state: Information about the current execution context

        Returns:
            Reshaped tensor
        """
        data = tf.reshape(data, self.shape)
        return data
