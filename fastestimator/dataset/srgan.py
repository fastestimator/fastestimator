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
"""The scripts assumes that imagenet dataset has already been downloaded.
Expectation is that, Imagenet directory has 'train'  and 'val' subdirectory having 1000 subdirectory each representing classes.
Creates the train and val csv needed to train srgan and srresnet.
"""

from glob import glob
import cv2
import os 
import PIL
from PIL import Image
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd

def _generate_crop_samples(in_tup):
    imgpath, path_hr, path_lr = in_tup
    filename = os.path.basename(imgpath)
    filename = os.path.splitext(filename)
    img = Image.open(imgpath)    
    width, height = img.size
    if height >= 96 and width >= 96:
        crp_ht = height - 96 + 1
        crp_wd = width - 96 + 1
        
        crp_y1 = np.random.randint(low=0, high=crp_ht)
        crp_x1 = np.random.randint(low=0, high=crp_wd)
        
        crp_img = img.crop((crp_x1, crp_y1, crp_x1+96, crp_y1+96))
        # fn = filename[0]+'_hr'+filename[1]
        fn = filename[0]+'_hr'+'.png'
        fn = os.path.join(path_hr,fn)
        crp_img.save(fn)
        
        cr_w, cr_h = crp_img.size
        
        crp_img_resize = crp_img.resize((cr_w//4, cr_h//4),resample=PIL.Image.BICUBIC )
        # fn = filename[0]+'_lr'+filename[1]
        fn = filename[0]+'_lr'+'.png'
        fn = os.path.join(path_lr,fn)
        crp_img_resize.save(fn)




def  _generate_samples(ext, path_imgnet):
    # generating samples for high resolution 96*96 and low resolution 24*24 imgs

    imgnet_samples = glob(os.path.join(path_imgnet, ext ,'*','*.JPEG'))
    if ext=='train': 
        selection_size = 360000
        imgnet_samples_idx = range(len(imgnet_samples))
        rand_indx =  np.random.choice(imgnet_samples_idx, size=selection_size,replace=False)

    else: 
        selection_size = 50000
        rand_indx = range(selection_size) 
    selected_imgnet = [imgnet_samples[sel_idx] for sel_idx in rand_indx]

    hr_ext = ext+'_hr'
    path_hr = os.path.join(path_imgnet,  hr_ext)
    path_hr_not_exists = not os.path.exists(path_hr)
    if not os.path.exists(path_hr):
        os.makedirs(path_hr)

    lr_ext = ext+'_lr'
    path_lr =os.path.join(path_imgnet, lr_ext)
    path_lr_not_exists = not os.path.exists(path_lr)
    if not os.path.exists(path_lr):
        os.makedirs(path_lr)

    path_lr_empty = os.path.exists(path_lr) and  len(os.listdir(path_lr))==0
    path_hr_empty = os.path.exists(path_hr) and len(os.listdir(path_hr))==0



    if path_hr_not_exists or path_hr_empty or path_lr_not_exists or path_lr_empty:
        num_cpu = mp.cpu_count()
        pool = Pool(processes=num_cpu)
        length = len(selected_imgnet)
        pool.map(_generate_crop_samples, zip(selected_imgnet, [path_hr]*length, [path_lr]*length))

    # images_lr = glob(os.path.join(path_imgnet, lr_ext,'*.JPEG'))
    # images_hr = glob(os.path.join(path_imgnet, hr_ext,'*.JPEG'))
    images_lr = glob(os.path.join(path_imgnet, lr_ext,'*.png'))
    images_hr = glob(os.path.join(path_imgnet, hr_ext,'*.png'))
    
    csv_path = _create_csv(images_lr, images_hr, ext, path_imgnet)
    
    return csv_path
        

def _create_csv(images_lr, images_hr, ext, path_imgnet):
    images_lr = sorted(images_lr)
    images_hr = sorted(images_hr)
    frame = { 'lowres':images_lr,'highres':images_hr}
    df = pd.DataFrame(frame)
    fn = 'super_res_' + ext +'.csv'
    df.to_csv(os.path.join(path_imgnet, fn), index=False)
    return os.path.join(path_imgnet, fn)


def load_data(path_imgnet=None):
    
    assert path_imgnet is not None, 'path_imgnet should have valid directory where imagenet dataset has been downloaded' 
    train_csv  = _generate_samples('train', path_imgnet)
    val_csv = _generate_samples('val', path_imgnet)
    return train_csv, val_csv, path_imgnet






