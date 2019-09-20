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
import multiprocessing as mp
import os
import tempfile

import pandas as pd
import tensorflow as tf
from PIL import Image


def write_images_serial(start_idx, end_idx, data, image_path, mode):
    for i in range(start_idx, end_idx):
        img = Image.fromarray(data[i])
        img.save(os.path.join(image_path, "{}_{}.png".format(mode, i)))


def write_images_parallel(x, image_path, mode):
    num_cpu = mp.cpu_count()
    num_example = x.shape[0]
    example_per_cpu = num_example // num_cpu
    processes = []
    for rank in range(num_cpu):
        start_idx = rank * example_per_cpu
        if rank == num_cpu - 1:
            end_idx = num_example
        else:
            end_idx = start_idx + example_per_cpu
        processes.append(mp.Process(target=write_images_serial, args=(start_idx, end_idx, x, image_path, mode)))
    for p in processes:
        p.start()
        p.join()


def write_csv(y, csv_path, mode):
    x_names = []
    y_names = []
    num_example = len(y)
    for idx in range(num_example):
        x_names.append(os.path.join("image", "{}_{}.png".format(mode, idx)))
        y_names.append(y[idx])
    df = pd.DataFrame(data={'x': x_names, 'y': y_names})
    df.to_csv(csv_path, index=False)


def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "Mnist")
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, "image")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_csv_path = os.path.join(path, "train.csv")
    eval_csv_path = os.path.join(path, "eval.csv")
    if not os.path.exists(image_path):
        print("writing image data to {}".format(image_path))
        os.makedirs(image_path)
        write_images_parallel(x_train, image_path, "train")
        write_images_parallel(x_test, image_path, "eval")
    if not os.path.exists(train_csv_path):
        write_csv(y_train, train_csv_path, "train")
    if not os.path.exists(eval_csv_path):
        write_csv(y_test, eval_csv_path, "eval")
    return train_csv_path, eval_csv_path, path
