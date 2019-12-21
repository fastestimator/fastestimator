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
"""Download USPS Dataset."""
import gzip
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wget
from PIL import Image

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _write_images_serial(start_idx, end_idx, data, image_path, mode):
    for idx in range(start_idx, end_idx):
        curr_data = data[idx]
        curr_data = (curr_data - curr_data.min()) / max(curr_data.max() - curr_data.min(), 1e-8) * 255
        img = Image.fromarray(curr_data.astype(np.uint8))
        img.save(os.path.join(image_path, '{}_{}.png'.format(mode, idx)))


def _write_images_parallel(img, image_path, mode):
    num_cpu = mp.cpu_count()
    num_example = img.shape[0]
    example_per_cpu = num_example // num_cpu
    processes = []
    for rank in range(num_cpu):
        start_idx = rank * example_per_cpu
        if rank == num_cpu - 1:
            end_idx = num_example
        else:
            end_idx = start_idx + example_per_cpu
        processes.append(mp.Process(target=_write_images_serial, args=(start_idx, end_idx, img, image_path, mode)))
    for process in processes:
        process.start()
        process.join()


def _create_csv(label, csv_path, mode):
    x_names = []
    y_names = []
    num_example = len(label)
    for idx in range(num_example):
        x_names.append(os.path.join('image', '{}_{}.png'.format(mode, idx)))
        y_names.append(label[idx])
    df = pd.DataFrame(data={'x': x_names, 'y': y_names})
    df.to_csv(csv_path, index=False)
    print("Data summary is saved at {}".format(csv_path))


# https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
def extract_images_labels(filename):
    print('Extracting', filename)
    with gzip.open(filename, 'rb') as f:
        raw_data = f.read().split()
    data = np.asarray([raw_data[start:start + 257] for start in range(0, len(raw_data), 257)], dtype=np.float32)
    images_vec = data[:, 1:]
    images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
    labels = data[:, 0].astype(int)
    #     import pdb; pdb.set_trace()
    return images, labels


def load_data(path=None):
    """Download the USPS dataset to local storage, if not already downloaded. This will generate 2 csv files
    (train, eval), which contain all the path information.

    Args:
        path (str, optional): The path to store the MNIST data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (train_csv, eval_csv, path) tuple, where
        
        * **train_csv** (str) -- Path to train csv file, containing the following columns:
        
            * x (str): Image directory relative to the returned path.
            * y (int): Label which indicates the number shown in image.
            
        * **eval_csv** (str) -- Path to test csv file, containing the same columns as train_csv.
        * **path** (str) -- Path to data directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'USPS')
    else:
        path = os.path.join(os.path.abspath(path), 'USPS')
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, 'image')
    train_csv = os.path.join(path, 'train.csv')
    eval_csv = os.path.join(path, 'eval.csv')

    # download data to memory
    train_compressed_path = os.path.join(path, "zip.train.gz")
    test_compressed_path = os.path.join(path, "zip.test.gz")

    if not os.path.exists(train_compressed_path):
        print("Downloading train data to {}".format(path))
        wget.download('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz', path, bar=bar_custom)
    train_images, train_labels = extract_images_labels(train_compressed_path)

    if not os.path.exists(test_compressed_path):
        print("Downloading test data to {}".format(path))
        wget.download('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz', path, bar=bar_custom)
    test_images, test_labels = extract_images_labels(test_compressed_path)

    # write to disk
    if not os.path.exists(image_path):
        print("Writing image data to {}".format(image_path))
        os.makedirs(image_path)
        _write_images_parallel(train_images, image_path, 'train')
        _write_images_parallel(test_images, image_path, 'eval')

    # generate csv
    if not os.path.exists(train_csv):
        _create_csv(train_labels, train_csv, 'train')
    if not os.path.exists(eval_csv):
        _create_csv(test_labels, eval_csv, 'eval')

    return train_csv, eval_csv, path
