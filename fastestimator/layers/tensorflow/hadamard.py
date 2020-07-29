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

    Raises:
        ValueError: If `code_length` is invalid.
    """
    heads: Union[List[layers.Dense], layers.Dense]

    def __init__(self, n_classes: int, code_length: Optional[int] = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if code_length is None:
            code_length = max(16, 1 << (n_classes - 1).bit_length())
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}")
        self.code_length = code_length
        self.labels = None
        self.heads = []
        self._call_fn = None

    def get_config(self) -> Dict[str, Any]:
        return {'n_classes': self.n_classes, 'code_length': self.code_length}

    def build(self, input_shape: Union[Tuple[int, int], List[Tuple[int, int]]]) -> None:
        single_input = not isinstance(input_shape, list)
        input_shape = to_list(input_shape)
        batch_size = input_shape[0][0]
        if len(input_shape) > self.code_length:
            raise ValueError(f"Too many input heads {len(input_shape)} for the given code length {self.code_length}.")
        head_sizes = [self.code_length // len(input_shape) for _ in range(len(input_shape))]
        head_sizes[0] = head_sizes[0] + self.code_length - sum(head_sizes)
        for idx, shape in enumerate(input_shape):
            if len(shape) != 2:
                raise ValueError("ErrorCorrectingCode layer requires input like (batch, m) or [(batch, m), ...]")
            if shape[0] != batch_size:
                raise ValueError("Inputs to ErrorCorrectingCode layer must have the same batch size")
            self.heads.append(layers.Dense(units=head_sizes[idx]))
        self.labels = tf.transpose(tf.convert_to_tensor(hadamard(self.code_length)[:self.n_classes], dtype=tf.float32))
        # Spare extra operations when they're not needed
        if single_input:
            self.heads = self.heads[0]
            self._call_fn = self._single_head_call
        else:
            self._call_fn = self._multi_head_call

    def _single_head_call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.heads(x)
        x = tf.tanh(x)
        x = tf.matmul(x, self.labels) + self.code_length
        x = tf.math.divide(x, tf.reshape(tf.reduce_sum(x, axis=1), (-1, 1)))
        return x

    def _multi_head_call(self, x: List[tf.Tensor]) -> tf.Tensor:
        x = [head(tensor) for head, tensor in zip(self.heads, x)]
        x = tf.concat(x, axis=-1)
        x = tf.tanh(x)
        x = tf.matmul(x, self.labels) + self.code_length
        x = tf.math.divide(x, tf.reshape(tf.reduce_sum(x, axis=1), (-1, 1)))
        return x

    def call(self, x: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        return self._call_fn(x)
