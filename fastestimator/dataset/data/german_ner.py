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
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fastestimator.dataset.numpy_dataset import NumpyDataset


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
        if line[0] != '#':
            line = line.split()
            if len(line) > 2 and line[2] != 'O':
                words.append(line[1])
                tags.append(line[2])
                word_vocab.add(line[1])
                label_vocab.add(line[2])
            else:
                sentences.append(" ".join([s for s in words]))
                labels.append([t for t in tags])
                words.clear()
                tags.clear()
    sentences = list(filter(None, sentences))
    labels = list(filter(None, labels))
    return sentences[:10000], labels[:10000], word_vocab, label_vocab


def load_data(root_dir: Optional[str] = None) -> Tuple[NumpyDataset, NumpyDataset, Set[str], Set[str]]:
    """Load and return the GermEval dataset.

    Dataset from GermEval 2014 contains 31,000 sentences corresponding to over 590,000 tokens from German wikipedia
    and News corpora. The sentence is encoded as one token per line with information provided in tab-seprated columns.
    Sourced from https://sites.google.com/site/germeval2014ner/data

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, eval_data, train_vocab, label_vocab)
    """
    url = 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1'
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'GermEval')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'GermEval')
    os.makedirs(root_dir, exist_ok=True)

    data_path = os.path.join(root_dir, 'de_ner.tsv')
    data_folder_path = os.path.join(root_dir, 'germeval')

    if not os.path.exists(data_folder_path):
        # download
        if not os.path.exists(data_path):
            print("Downloading data to {}".format(root_dir))
            stream = requests.get(url, stream=True)  # python wget does not work
            total_size = int(stream.headers.get('content-length', 0))
            block_size = 128  # 1 MB
            progress = tqdm(total=total_size, unit='B', unit_scale=True)
            with open(data_path, 'wb') as outfile:
                for data in stream.iter_content(block_size):
                    progress.update(len(data))
                    outfile.write(data)
            progress.close()

    x, y, x_vocab, y_vocab = get_sentences_and_labels(data_path)

    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.array(x_train)
    x_eval = np.array(x_eval)
    y_train = np.array(y_train)
    y_eval = np.array(y_eval)
    train_data = NumpyDataset({"x": x_train, "y": y_train})
    eval_data = NumpyDataset({"x": x_eval, "y": y_eval})
    return train_data, eval_data, x_vocab, y_vocab
