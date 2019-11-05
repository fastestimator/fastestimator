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



class TensorFilter(TensorOp):
    """Base class for all the filtering TensorOps.

    Args:
         inputs: Name of the key in the dataset that is to be filtered.
         mode: Mode that the filter acts on. This could be either "train" or "eval".
    """
    def __init__(self, inputs, mode="train"):
        super().__init__(inputs=inputs, mode=mode)

    def forward(self, data, state):
        return tf.constant(True)


class ScalarFilter(TensorFilter):
    """Class for performing filtering on dataset based on scalar values.

    Args:
        inputs: Name of the key in the dataset that is to be filtered.
        filter_value: The values in the dataset that are to be filtered.
        keep_prob: The probability of keeping the example.
        mode: mode that the filter acts on.
    """
    def __init__(self, inputs, filter_value, keep_prob, mode="train"):
        super().__init__(inputs=inputs, mode=mode)
        self.filter_value = filter_value
        self.keep_prob = keep_prob
        self._verify_inputs()

    def _verify_inputs(self):
        assert isinstance(self.inputs, str), "ScalarFilter only accepts single string input"
        self.filter_value = to_list(self.filter_value)
        self.keep_prob = to_list(self.keep_prob)
        assert len(self.filter_value) == len(self.keep_prob), "filter_value and keep_prob must be same length"

    def forward(self, data, state):
        """Filters the data based on the scalar filter_value.

        Args:
            data: Data to be filtered.
            state: Information about the current execution context.

        Returns:
            Tensor containing filtered data.
        """
        pass_filter = tf.constant(True)
        for filter_value, keep_prob in zip(self.filter_value, self.keep_prob):
            pass_current_filter = tf.cond(tf.equal(data, filter_value),
                                          lambda: tf.greater(keep_prob, tf.random.uniform([])),
                                          lambda: tf.constant(True))
            pass_filter = tf.logical_and(pass_filter, pass_current_filter)
        return pass_filter
