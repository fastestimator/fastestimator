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
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import wget

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None, seq_length: int = 100) -> Tuple[NumpyDataset, List[str]]:
    """Load and return the Shakespeare dataset.

    Shakespeare dataset is a collection of texts written by Shakespeare.
    Sourced from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        seq_length: Length of data sequence.

    Returns:
        (train_data, vocab)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'Shakespeare')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'Shakespeare')
    os.makedirs(root_dir, exist_ok=True)

    file_path = os.path.join(root_dir, 'shakespeare.txt')
    download_link = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

    if not os.path.exists(file_path):
        # Download
        print("Downloading data: {}".format(file_path))
        wget.download(download_link, file_path, bar=bar_custom)

    with open(file_path, 'rb') as f:
        text_data = f.read().decode(encoding='utf-8')

    # Build dictionary from training data
    vocab = sorted(set(text_data))
    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_data = [char2idx[c] for c in text_data] + [0] * (seq_length - len(text_data) % seq_length)
    text_data = np.array(text_data).reshape(-1, seq_length)
    train_data = NumpyDataset(data={"x": text_data})
    return train_data, vocab
