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
"""
Download montgomery  dataset from  http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
"""
import os
import tempfile
from glob import glob
import zipfile
import shutil

import numpy as np
import pandas as pd
import wget

def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", 'MONTGOMERY')
    if not os.path.exists(path):
        os.makedirs(path)
    csv_path = os.path.join(path, "montgomery.csv")
    data_compressed_path = os.path.join(path, 'NLM-MontgomeryCXRSet.zip')
    extract_folder_path = os.path.join(path, 'MontgomerySet')
    if not os.path.exists(data_compressed_path):
        print("Downloading data...")
        wget.download('http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip', path)
    if not os.path.exists(extract_folder_path):
        print(" ")
        print("Extracting data...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)
        shutil.move(os.path.join(extract_folder_path, "ManualMask", "leftMask"), os.path.join(extract_folder_path, "leftMask"))
        shutil.move(os.path.join(extract_folder_path, "ManualMask", "rightMask"), os.path.join(extract_folder_path, "rightMask"))
    if not os.path.exists(csv_path):
        img_list = glob(os.path.join(extract_folder_path, "CXR_png", '*.png'))
        df = pd.DataFrame(data={'image': img_list})
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, path))
        df['mask_left'] = df['image'].str.replace('CXR_png', 'leftMask')
        df['mask_right'] = df['image'].str.replace('CXR_png', 'rightMask')
        df.to_csv(csv_path, index=False)
    return csv_path, path
