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
import gzip
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import wget
from PIL import Image

from fastestimator.dataset.labeled_dir_dataset import LabeledDirDatasets
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


class USPSDataset(LabeledDirDatasets):
    """
    Download the usps dataset, if not already downloaded.

    Args:
        root_dir: The path to store the usps data.
    """
    def __init__(self, root_dir: str = None):
        home = str(Path.home())

        if root_dir is None:
            root_dir = os.path.join(home, 'fastestimator_data', 'USPS')
        else:
            root_dir = os.path.join(os.path.abspath(root_dir), 'USPS')
        os.makedirs(root_dir, exist_ok=True)

        # download data to memory
        train_compressed_path = os.path.join(root_dir, "zip.train.gz")
        test_compressed_path = os.path.join(root_dir, "zip.test.gz")

        if not os.path.exists(train_compressed_path):
            print("Downloading train data to {}".format(root_dir))
            wget.download('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
                          root_dir,
                          bar=bar_custom)
        train_images, train_labels = self.extract_images_labels(train_compressed_path)

        if not os.path.exists(test_compressed_path):
            print("Downloading test data to {}".format(root_dir))
            wget.download('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz',
                          root_dir,
                          bar=bar_custom)
        test_images, test_labels = self.extract_images_labels(test_compressed_path)

        # write to disk
        self.write_data(train_images, train_labels, root_dir, "train")
        self.write_data(test_images, test_labels, root_dir, "eval")

        super().__init__(root_dir=root_dir, file_extension='.png')

    @staticmethod
    def write_image(data: Tuple[np.ndarray, str, int]):
        image, path, idx = data
        image = (image - image.min()) / max(image.max() - image.min(), 1e-8) * 255
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(path, 'img_{}.png'.format(idx)))

    @staticmethod
    def write_data(images: np.ndarray, labels: np.ndarray, root_dir: str, mode: str):
        base_path = os.path.join(root_dir, mode)
        if not os.path.exists(base_path):
            print("Writing image data to {}".format(base_path))
            os.makedirs(base_path)
            for i in range(min(labels), max(labels) + 1):
                os.makedirs(os.path.join(base_path, "{}".format(i)))
            with Pool(os.cpu_count()) as p:
                p.map(USPSDataset.write_image,
                      zip(images, map(lambda l: os.path.join(base_path, "{}".format(l)), labels), range(len(labels))))

    @staticmethod
    def extract_images_labels(filename):
        # https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
        print('Extracting', filename)
        with gzip.open(filename, 'rb') as f:
            raw_data = f.read().split()
        data = np.asarray([raw_data[start:start + 257] for start in range(0, len(raw_data), 257)], dtype=np.float32)
        images_vec = data[:, 1:]
        images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
        labels = data[:, 0].astype(int)
        return images, labels


if __name__ == "__main__":
    ds = USPSDataset()
