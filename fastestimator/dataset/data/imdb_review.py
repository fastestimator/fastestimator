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
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from fastestimator.dataset.numpy_dataset import NumpyDataset


def pad(input_list: List[int], padding_size: int, padding_value: int) -> List[int]:
    """Pad an input_list to a given size.

    Args:
        input_list: The list to be padded.
        padding_size: The desired length of the returned list.
        padding_value: The value to be inserted for padding.

    Returns:
        `input_list` with `padding_value`s appended until the `padding_size` is reached.
    """
    return input_list + [padding_value] * abs((len(input_list) - padding_size))


def load_data(max_len: int, vocab_size: int) -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the IMDB Movie review dataset.

    This dataset contains 25,000 reviews labeled by sentiments (either positive or negative).

    Args:
        max_len: Maximum desired length of an input sequence.
        vocab_size: Vocabulary size to learn word embeddings.

    Returns:
        (train_data, eval_data)
    """
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    # pad the sequences to max length
    x_train = np.array([pad(x, max_len, 0) for x in x_train])
    x_eval = np.array([pad(x, max_len, 0) for x in x_eval])

    train_data = NumpyDataset({"x": x_train, "y": y_train})
    eval_data = NumpyDataset({"x": x_eval, "y": y_eval})
    return train_data, eval_data
