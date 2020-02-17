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
"""Download MNIST Dataset."""
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from PIL import Image


def _write_images_serial(start_idx, end_idx, data, image_path, mode):
    for idx in range(start_idx, end_idx):
        img = Image.fromarray(data[idx])
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


def load_data(path=None):
    """Download the MNIST dataset to local storage, if not already downloaded. This will generate 2 csv files
    (train, eval), which contain all the path information.

    Args:
        path (str, optional): The path to store the MNIST data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (train_csv, eval_csv, path) tuple, where
        
        * **train_csv** (str) -- Path to train csv file, containing the following columns:
        
            * x (str): Image directory relative to the returned path.
            * y (int): Label indicating the number shown in the image.
        
        * **eval_csv** (str) -- Path to test csv file, containing the same columns as train_csv.
        * **path** (str) -- Path to data directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'MNIST')
    else:
        path = os.path.join(os.path.abspath(path), 'MNIST')
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, 'image')
    train_csv = os.path.join(path, 'train.csv')
    eval_csv = os.path.join(path, 'eval.csv')

    # download data to memory
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

    # write to disk
    if not os.path.exists(image_path):
        print("Writing image data to {}".format(image_path))
        os.makedirs(image_path)
        _write_images_parallel(x_train, image_path, 'train')
        _write_images_parallel(x_eval, image_path, 'eval')

    # generate csv
    if not os.path.exists(train_csv):
        _create_csv(y_train, train_csv, 'train')
    if not os.path.exists(eval_csv):
        _create_csv(y_eval, eval_csv, 'eval')

    return train_csv, eval_csv, path
