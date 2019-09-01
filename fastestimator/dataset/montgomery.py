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

    # montgomery_img = os.path.join(path, 'MontgomerySet', 'CXR_png')
    # montgomery_leftmask_img = os.path.join(path, 'MontgomerySet', 'ManualMask', 'leftMask')
    # montgomery_rightmask_img = os.path.join(path, 'MontgomerySet', 'ManualMask', 'rightMask')
    # montgomery_combinedmask_img = os.path.join(path, 'MontgomerySet', 'combinedMask')

    # montgomery_img_list = glob(os.path.join(montgomery_img, '*.png'))
    # montgomery_leftmask_list = glob(os.path.join(montgomery_leftmask_img, '*.png'))
    # montgomery_rightmask_list = glob(os.path.join(montgomery_rightmask_img, '*.png'))

    # if not os.listdir(montgomery_combinedmask_img):
    #     for leftmaskpath, rightmaskpath in zip(montgomery_leftmask_list, montgomery_rightmask_list):
    #         lm_base = os.path.basename(leftmaskpath)
    #         rm_base = os.path.basename(rightmaskpath)
    #         if lm_base == rm_base:
    #             leftmask = cv2.imread(leftmaskpath, cv2.IMREAD_GRAYSCALE)
    #             rightmask = cv2.imread(rightmaskpath, cv2.IMREAD_GRAYSCALE)

    #             mask = np.maximum(leftmask, rightmask)
    #             cv2.imwrite(os.path.join(montgomery_combinedmask_img, lm_base), mask)

    # montgomery_combinedmask_list = glob(os.path.join(montgomery_combinedmask_img, '*.png'))
    # bn_2_imgpath = {os.path.basename(i): i for i in montgomery_img_list}
    # bn_2_maskpath = {os.path.basename(i): i for i in montgomery_combinedmask_list}

    # train_cvs_path = os.path.join(path, 'train_image_mask.csv')
    # eval_cvs_path = os.path.join(path, 'eval_image_mask.csv')

    # df_all = pd.DataFrame({'basename': list(bn_2_imgpath.keys()), 'imgpath': list(bn_2_imgpath.values())})
    # df_all['mask'] = df_all.basename.map(bn_2_maskpath)
    # df_all.drop(['basename'], axis=1, inplace=True)
    # df_train = df_all[:100]
    # df_val = df_all[100:]
    # df_train.to_csv(train_cvs_path, index=False)
    # df_val.to_csv(eval_cvs_path, index=False)

    return data_csv, path
