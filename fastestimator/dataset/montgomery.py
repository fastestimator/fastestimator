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
"""Download Montgomery dataset from  http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip."""
import os
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(path=None):
    """Download the Montgomery dataset to local storage, if not already downloaded. This will generate a montgomery.csv
    file, which contains all the path information.

    Args:
        path (str, optional): The path to store the Montgomery data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (csv_path, path) tuple, where
        
        * **csv_path** (str) -- Path to the summary csv file, containing the following columns:
        
            * image (str): Image directory relative to the returned path.
            * mask_left (str): Left lung mask directory relative to the returned path.
            * mask_right (str): Right lung mask directory relative to the returned path.
            
        * **path** (str) -- Path to data directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'Montgomery')
    else:
        path = os.path.join(os.path.abspath(path), 'Montgomery')
    os.makedirs(path, exist_ok=True)

    csv_path = os.path.join(path, "montgomery.csv")
    data_compressed_path = os.path.join(path, 'NLM-MontgomeryCXRSet.zip')
    extract_folder_path = os.path.join(path, 'CXRSet')

    # download
    if not os.path.exists(data_compressed_path):
        print("Downloading data to {}".format(path))
        wget.download('http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip', path, bar=bar_custom)

    # extract
    if not os.path.exists(extract_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)
        os.rename(os.path.join(path, 'MontgomerySet'), extract_folder_path)
        os.rename(os.path.join(extract_folder_path, 'ManualMask', 'leftMask'),
                  os.path.join(extract_folder_path, 'leftMask'))
        os.rename(os.path.join(extract_folder_path, 'ManualMask', 'rightMask'),
                  os.path.join(extract_folder_path, 'rightMask'))

    # glob and generate csv
    if not os.path.exists(csv_path):
        img_list = glob(os.path.join(extract_folder_path, 'CXR_png', '*.png'))
        df = pd.DataFrame(data={'image': img_list})
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, path))
        df['image'] = df['image'].apply(os.path.normpath)
        df['mask_left'] = df['image'].str.replace('CXR_png', 'leftMask')
        df['mask_right'] = df['image'].str.replace('CXR_png', 'rightMask')
        df.to_csv(csv_path, index=False)

    return csv_path, path
