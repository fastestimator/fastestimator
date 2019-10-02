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


class Resize(TensorOp):
    """Preprocessing class for resizing the images.

    Args:
        size: Destination shape of the images.
        resize_method: One of resize methods provided by tensorflow to be used.
        inputs: Name of the key in the dataset that is to be filtered.
        outputs: Name of the key to be created/used in the dataset to store the results.
        mode: mode that the filter acts on.
    """
    def __init__(self, size, resize_method=tf.image.ResizeMethod.BILINEAR, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.size = size
        self.resize_method = resize_method

    def forward(self, data, state):
        """Resizes data tensor.

        Args:
            data: Tensor to be resized.
            state: Information about the current execution context.

        Returns:
            Resized tensor.
        """
        preprocessed_data = tf.image.resize(data, self.size, self.resize_method)
        return preprocessed_data
