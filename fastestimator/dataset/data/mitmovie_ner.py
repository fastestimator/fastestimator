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
from typing import List, Optional, Set, Tuple

import numpy as np
import wget

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def get_sentences_and_labels(path: str) -> Tuple[List[str], List[List[str]], Set[str], Set[str]]:
    """Combines tokens into sentences and create vocab set for train data and labels.

    For simplicity tokens with 'O' entity are omitted.

    Args:
        path: Path to the downloaded dataset file.

    Returns:
        (sentences, labels, train_vocab, label_vocab)
    """
    words, tags = [], []
    word_vocab, label_vocab = set(), set()
    sentences, labels = [], []
    data = open(path)
    for line in data:
        if line != '\n':
            line = line.split()
            words.append(line[1])
            tags.append(line[0])
            word_vocab.add(line[1])
            label_vocab.add(line[0])
        else:
            sentences.append(" ".join([s for s in words]))
            labels.append([t for t in tags])
            words.clear()
            tags.clear()
    sentences = list(filter(None, sentences))
    labels = list(filter(None, labels))
    return sentences, labels, word_vocab, label_vocab


def load_data(root_dir: Optional[str] = None) -> Tuple[NumpyDataset, NumpyDataset, Set[str], Set[str]]:
    """Load and return the MIT Movie dataset.

    MIT Movies dataset is a semantically tagged training and test corpus in BIO format. The sentence is encoded as one
    token per line with information provided in tab-seprated columns.
    Sourced from https://groups.csail.mit.edu/sls/downloads/movie/

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, eval_data, train_vocab, label_vocab)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'MITMovie')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'MITMovie')
    os.makedirs(root_dir, exist_ok=True)

    train_data_path = os.path.join(root_dir, 'engtrain.bio')
    test_data_path = os.path.join(root_dir, 'engtest.bio')
    files = [(train_data_path, 'https://groups.csail.mit.edu/sls/downloads/movie/engtrain.bio'),
             (test_data_path, 'https://groups.csail.mit.edu/sls/downloads/movie/engtest.bio')]

    for data_path, download_link in files:
        if not os.path.exists(data_path):
            # Download
            print("Downloading data: {}".format(data_path))
            wget.download(download_link, data_path, bar=bar_custom)

    x_train, y_train, x_vocab, y_vocab = get_sentences_and_labels(train_data_path)
    x_eval, y_eval, x_eval_vocab, y_eval_vocab = get_sentences_and_labels(test_data_path)
    x_vocab |= x_eval_vocab
    y_vocab |= y_eval_vocab
    x_train = np.array(x_train)
    x_eval = np.array(x_eval)
    y_train = np.array(y_train)
    y_eval = np.array(y_eval)
    train_data = NumpyDataset({"x": x_train, "y": y_train})
    eval_data = NumpyDataset({"x": x_eval, "y": y_eval})
    return train_data, eval_data, x_vocab, y_vocab
