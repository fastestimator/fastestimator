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
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from scipy.linalg import hadamard
from tensorflow.python.keras import layers

from fastestimator.util.util import to_list


class HadamardCode(layers.Layer):
    """A layer for applying an error correcting code to your outputs.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    See 'https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-
    robustness-of-deep-neural-networks'. Note that for best effectiveness, the model leading into this layer should be
    split into multiple independent chunks, whose outputs this layer can combine together in order to perform the code
    lookup.

    ```python
    # Use as a drop-in replacement for your softmax layer:
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(10, activation='softmax'))
    #   ----- vs ------
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(HadamardCode(10))
    ```

    ```python
    # Use to combine multiple feature heads for a final output (biggest adversarial hardening benefit):
    inputs = Input(input_shape)
    features = Dense(1024, activation='relu')(inputs)
    heads = [Dense(20)(features) for _ in range(5)]
    outputs = HadamardCode(10)(heads)
    model = Model(inputs, outputs)
    ```

    Args:
        n_classes: How many output classes to map onto.
        code_length: How long of an error correcting code to use. Should be a positive multiple of 2. If not provided,
            the smallest power of 2 which is >= `n_outputs` will be used, or 16 if the latter is larger.
        max_prob: The maximum probability that can be assigned to a class. For numeric stability this must be less than
            1.0. Intuitively it makes sense to keep this close to 1, but to get adversarial training benefits it should
            be noticeably less than 1, for example 0.95 or even 0.8.
        power: The power parameter to be used by Inverse Distance Weighting when transforming Hadamard class distances
            into a class probability distribution. A value of 1.0 gives an intuitive mapping to probabilities, but small
            values such as 0.25 appear to give slightly better adversarial benefits. Large values like 2 or 3 give
            slightly faster convergence at the expense of adversarial performance. Must be greater than zero.

    Raises:
        ValueError: If `code_length`, `max_prob`, or `power` are invalid.
    """
    heads: Union[List[layers.Dense], layers.Dense]

    def __init__(self, n_classes: int, code_length: Optional[int] = None, max_prob: float = 0.95,
                 power: float = 1.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        if code_length is None:
            code_length = max(16, 1 << (n_classes - 1).bit_length())
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}.")
        self.code_length = code_length
        if power <= 0:
            raise ValueError(f"power must be positive, but got {power}.")
        self.power = power
        if not 0.0 < max_prob < 1.0:
            raise ValueError(f"max_prob must be in the range (0, 1), but got {max_prob}")
        self.eps = self.code_length * math.pow((1.0 - max_prob) / (max_prob * (self.n_classes - 1)), 1 / self.power)
        self.labels = None
        self.heads = []
        self._call_fn = None

    def get_config(self) -> Dict[str, Any]:
        return {'n_classes': self.n_classes, 'code_length': self.code_length}

    def build(self, input_shape: Union[Tuple[int, int], List[Tuple[int, int]]]) -> None:
        single_input = not isinstance(input_shape, list)
        input_shape = to_list(input_shape)
        batch_size = input_shape[0][0]
        if len(input_shape) > self.code_length - 1:
            raise ValueError(f"Too many input heads {len(input_shape)} for the given code length {self.code_length}.")
        head_sizes = [self.code_length // len(input_shape) for _ in range(len(input_shape))]
        head_sizes[0] = head_sizes[0] + self.code_length - sum(head_sizes)
        head_sizes[0] = head_sizes[0] - 1  # We're going to cut off the 0th column from the code
        for idx, shape in enumerate(input_shape):
            if len(shape) != 2:
                raise ValueError("ErrorCorrectingCode layer requires input like (batch, m) or [(batch, m), ...]")
            if shape[0] != batch_size:
                raise ValueError("Inputs to ErrorCorrectingCode layer must have the same batch size")
            self.heads.append(layers.Dense(units=head_sizes[idx]))
        labels = hadamard(self.code_length)
        # Cut off 0th column b/c it's constant. It would also be possible to make the column sign alternate, but that
        # would break the symmetry between rows in the code.
        labels = labels[:self.n_classes, 1:]
        self.labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        # Spare extra operations when they're not needed
        if single_input:
            self.heads = self.heads[0]
            self._call_fn = self._single_head_call
        else:
            self._call_fn = self._multi_head_call

    def _single_head_call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.heads(x)
        x = tf.tanh(x)
        # Compute L1 distance
        x = tf.maximum(tf.reduce_sum(tf.abs(tf.expand_dims(x, axis=1) - self.labels), axis=-1), self.eps)
        # Inverse Distance Weighting
        x = 1.0 / tf.pow(x, self.power)
        x = tf.math.divide(x, tf.reshape(tf.reduce_sum(x, axis=-1), (-1, 1)))
        return x

    def _multi_head_call(self, x: List[tf.Tensor]) -> tf.Tensor:
        x = [head(tensor) for head, tensor in zip(self.heads, x)]
        x = tf.concat(x, axis=-1)
        x = tf.tanh(x)
        # Compute L1 distance
        x = tf.maximum(tf.reduce_sum(tf.abs(tf.expand_dims(x, axis=1) - self.labels), axis=-1), self.eps)
        # Inverse Distance Weighting
        x = 1.0 / tf.pow(x, self.power)
        x = tf.math.divide(x, tf.reshape(tf.reduce_sum(x, axis=-1), (-1, 1)))
        return x

    def call(self, x: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        return self._call_fn(x)
