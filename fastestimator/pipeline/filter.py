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

class TensorFilter:
    """
    An abstract class for data filter
    """
    def __init__(self, mode="train"):
        self.mode = mode

    def filter_fn(self, dataset):
        return tf.constant(True)


class ScalarFilter(TensorFilter):
    """
    Class for performing filtering on dataset based on scalar values.

    Args:
        feature_name: Name of the key in the dataset that is to be filtered
        filter_value: The values in the dataset that are to be filtered.
        keep_prob: The probability of keeping the example
        mode: mode that the filter acts on
    """
    def __init__(self, inputs, filter_value, keep_prob, mode="train"):
        self.inputs = inputs
        self.filter_value = filter_value
        self.keep_prob = keep_prob
        self.mode = mode
        self._verify_inputs()

    def _verify_inputs(self):
        self.inputs = self._convert_to_list(self.inputs)
        self.filter_value = self._convert_to_list(self.filter_value)
        self.keep_prob = self._convert_to_list(self.keep_prob)
        assert len(self.inputs) == len(self.filter_value) == len(self.keep_prob)

    def _convert_to_list(self, data):
        if not isinstance(data, list):
            data = [data]
        return data

    def filter_fn(self, dataset):
        pass_filter = tf.constant(True)
        for inp, filter_value, keep_prob in zip(self.inputs, self.filter_value, self.keep_prob):
            pass_current_filter = tf.cond(tf.equal(tf.reshape(dataset[inp], []), filter_value),
                                    lambda: tf.greater(keep_prob, tf.random.uniform([])),
                                    lambda: tf.constant(True))
            pass_filter = tf.logical_and(pass_filter, pass_current_filter)
        return pass_filter