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
"""Download omniglot dataset from https://github.com/brendenlake/omniglot/.
"""
import os
import zipfile
from pathlib import Path

import pandas as pd
import wget
from glob import glob
import numpy as np

import cv2

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress

# Randomly selected lanuages for the test dataset from evaluation set
TEST_LANGUAGES = [
    'Kannada', 'Tengwar', 'Aurek-Besh', 'Sylheti', 'Avesta', 'Glagolitic', 'Manipuri', 'Keble', 'Gurmukhi', 'Oriya'
]

# Number of images available for each character
IMAGES_PER_CHAR = 20


def _download_data(link, data_path, idx, total_idx):
    if not os.path.exists(data_path):
        print("Downloading data to {}, file: {} / {}".format(data_path, idx + 1, total_idx))
        wget.download(link, data_path, bar=bar_custom)


def load_data(path=None):
    """Download the Omniglot dataset to local storage.
    Args:
        path (str, optional): The path to store the  data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.
    Returns:
        tuple: (train_path, eval_path, path) tuple, where
        
        * **train_path** (str) -- Path to the train data folder
        
        * **train_path** (str) -- Path to the evaluation data folder.
    """
    if path is None:
        path = os.path.join(str(Path.home()), 'fastestimator_data', 'Omniglot')
    else:
        path = os.path.join(os.path.abspath(path), 'Omniglot')
    os.makedirs(path, exist_ok=True)

    #download data
    links = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    data_paths = [os.path.join(path, "images_{}.zip".format(x)) for x in ["background", "evaluation"]]
    for idx, (link, data_path) in enumerate(zip(links, data_paths)):
        _download_data(link, data_path, idx, len(links))

    #extract data
    image_extracted_path = os.path.join(path, 'images')
    if not os.path.exists(image_extracted_path):
        for idx, data_path in enumerate(data_paths):
            print("Extracting {}, file {} / {}".format(data_path, idx + 1, len(links)))
            with zipfile.ZipFile(data_path, 'r') as zip_file:
                zip_file.extractall(path)

    train_path = os.path.join(path, 'images_background')
    eval_path = os.path.join(path, 'images_evaluation')

    return train_path, eval_path


def load_eval_data(path, lang_list=TEST_LANGUAGES, is_test=False):
    """Load images for evaluation.
    Args:
        path (str): Path to evaluation folder.
        lang_list (list, optional): List of languages in test dataset, defaults to TEST_LANGUAGES
        is_test (bool, optional): whether to generate images for test dataset, defaults to False.
    Returns:
        img_list (list): List of images belonging to each language.
    """
    if not is_test:
        alphabet_list = os.listdir(path)
        lang_list = [i for i in alphabet_list if i not in lang_list]

    img_list = []
    for alphabet in lang_list:
        characters = os.listdir(os.path.join(path, alphabet))

        char_img_list = []
        for character in characters:
            img_paths = glob(os.path.join(path, alphabet, character, "*.png"))
            imgs = [np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1) for i in img_paths]
            char_img_list.append(imgs)

        img_list.append(char_img_list)

    return img_list


def get_batch(path, batch_size=128, is_train=True, test_lang_list=TEST_LANGUAGES):
    """Data generator for training and validation 
    Args:
        path (str): Path to folder containing the images.
        batch_size (int, optional): batch size, defaults to 128.
        is_train (bool, optional): whether to generate images for training or validation, defaults to True.
        test_lang_list (list, optional): List of languages in test dataset, defaults to TEST_LANGUAGES.
    Returns:
        (dict): Numpy arrays for the pair of images and label specifying whether the image pair belongs to same or different characters.
    """
    img_list = []

    alphabet_list = os.listdir(path)
    if not is_train:
        alphabet_list = [i for i in alphabet_list if i not in test_lang_list]

    for alphabet in alphabet_list:
        for character in os.listdir(os.path.join(path, alphabet)):
            img_paths = glob(os.path.join(path, alphabet, character, "*.png"))
            imgs = [np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1) for i in img_paths]
            img_list.append(imgs)

    list_length = len(img_list)

    while True:
        categories = np.random.choice(list_length, size=batch_size, replace=False)
        counter = 0

        for i in categories:
            img_1 = img_list[i][np.random.choice(IMAGES_PER_CHAR)]
            if counter < batch_size // 2:
                img_2 = img_list[i][np.random.choice(IMAGES_PER_CHAR)]
                target = 1
            else:
                img_2 = img_list[np.random.choice(np.delete(np.arange(list_length),
                                                            i))][np.random.choice(IMAGES_PER_CHAR)]
                target = 0

            counter = counter + 1
            yield {"x_a": np.array(img_1), "x_b": np.array(img_2), "y": [target]}


def one_shot_trial(img_list, N):
    """Generates one-shot trials 
    Args:
        img_list (list): List of images of aset of characters 
        N (int): Value of N for calculating N-way one-shot accuracy.
    Returns:
        (tuple): tuple of image pairs for one-shot verification
    """
    img_1_list = []
    img_2_list = []

    num_chars = len(img_list)
    index = np.random.choice(num_chars)
    base_image_index = np.random.choice(IMAGES_PER_CHAR, size=2, replace=False)

    base_image = img_list[index][base_image_index[0]]

    img_1_list.append(base_image)
    img_2_list.append(img_list[index][base_image_index[1]])

    if N > num_chars:
        test_indices = np.random.choice(np.delete(np.arange(num_chars), index), size=num_chars - 1, replace=False)
    else:
        test_indices = np.random.choice(np.delete(np.arange(num_chars), index), size=N - 1, replace=False)

    for i in test_indices:
        img_1_list.append(base_image)
        img_2_list.append(img_list[i][base_image_index[1]])

    return (np.array(img_1_list, dtype=np.float32) / 255, np.array(img_2_list, dtype=np.float32) / 255)
