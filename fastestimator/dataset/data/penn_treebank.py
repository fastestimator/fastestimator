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


def load_data(root_dir: Optional[str] = None,
              seq_length: int = 64) -> Tuple[NumpyDataset, NumpyDataset, NumpyDataset, List[str]]:
    """Load and return the Penn TreeBank dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        seq_length: Length of data sequence.

    Returns:
        (train_data, eval_data, test_data, vocab)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'PennTreeBank')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'PennTreeBank')
    os.makedirs(root_dir, exist_ok=True)

    train_data_path = os.path.join(root_dir, 'ptb.train.txt')
    eval_data_path = os.path.join(root_dir, 'ptb.valid.txt')
    test_data_path = os.path.join(root_dir, 'ptb.test.txt')

    files = [(train_data_path, 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'),
             (eval_data_path, 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'),
             (test_data_path, 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt')]

    texts = []
    for data_path, download_link in files:
        if not os.path.exists(data_path):
            # Download
            print("Downloading data: {}".format(data_path))
            wget.download(download_link, data_path, bar=bar_custom)

        text = []
        with open(data_path, 'r') as f:
            for line in f:
                text.extend(line.split() + ['<eos>'])
        texts.append(text)

    # Build dictionary from training data
    vocab = sorted(set(texts[0]))
    word2idx = {u: i for i, u in enumerate(vocab)}

    #convert word to index and split the sequences and discard the last incomplete sequence
    data = [[word2idx[word] for word in text[:-(len(text) % seq_length)]] for text in texts]
    x_train, x_eval, x_test = [np.array(d).reshape(-1, seq_length) for d in data]

    train_data = NumpyDataset(data={"x": x_train})
    eval_data = NumpyDataset(data={"x": x_eval})
    test_data = NumpyDataset(data={"x": x_test})
    return train_data, eval_data, test_data, vocab
