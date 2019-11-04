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
from fastestimator.op import TensorOp
from fastestimator.util.util import to_list


class Pad(TensorOp):
    """Pad data to target padded shape

    Args:
        padded_shape: target padded shape
        inputs: Name of the key in the dataset that is to be filtered.
        outputs: Name of the key to be created/used in the dataset to store the results.
        mode: mode that the filter acts on.
    """
    def __init__(self, padded_shape, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        padded_shape = to_list(padded_shape)
        self.padded_shape = tf.constant(padded_shape)

    def forward(self, data, state):
        """Pad data to target padded shape.

        Args:
            data: Data to be scaled.
            state: Information about the current execution context.

        Returns:
            padded data tensor
        """
        data = to_list(data)
        for idx, elem in enumerate(data):
            data_shape = tf.shape(elem)
            padding = self.padded_shape - data_shape
            padding = tf.transpose(tf.stack([tf.zeros_like(padding), padding]))
            data[idx] = tf.pad(elem, padding)
        return data
