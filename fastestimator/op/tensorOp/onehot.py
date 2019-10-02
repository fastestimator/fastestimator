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


class Onehot(TensorOp):
    """Preprocessing class for converting categorical labels to onehot encoding.

    Args:
        num_dim: Number of dimensions of the labels.
        inputs: Name of the key in the dataset that is to be filtered.
        outputs: Name of the key to be created/used in the dataset to store the results.
        mode: mode that the filter acts on.
    """
    def __init__(self, num_dim, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_dim = num_dim

    def forward(self, data, state):
        """Transforms categorical labels to onehot encodings.

        Args:
            data: Data to be preprocessed.
            state: Information about the current execution context.

        Returns:
            Transformed labels.
        """
        data = tf.cast(data, tf.int32)
        data = tf.one_hot(data, self.num_dim)
        return data
