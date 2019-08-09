import numpy as np
import os
import wget
from zipfile import ZipFile
import tempfile
import cv2
import pandas as pd 
from glob import glob 




def load_and_set_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(),'FE_MONTGOMERY')
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'MontgomerySet')) or not os.listdir(os.path.join(path,'MontgomerySet')):
        if not os.path.exists(os.path.join(path,'NLM-MontgomeryCXRSet.zip')):
            wget.download('http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip', path)
        print(f'extracting data in {path} ....')
        zippath = os.path.join(path,'NLM-MontgomeryCXRSet.zip')
        zipfile = ZipFile(zippath)
        zipfile.extractall(path)
    if 'combinedMask' not in os.listdir(os.path.join(path,'MontgomerySet')):
        os.mkdir(os.path.join(path,'MontgomerySet','combinedMask'))

    MONTGOMERY_IMG = os.path.join(path, 'MontgomerySet', 'CXR_png')
    MONTGOMERY_LEFTMASK_IMG = os.path.join(path,'MontgomerySet','ManualMask','leftMask' )
    MONTGOMERY_RIGHTMASK_IMG = os.path.join(path,'MontgomerySet','ManualMask','rightMask' )
    MONTGOMERY_COMBINEDMASK_IMG = os.path.join(path,'MontgomerySet','combinedMask')

    montgomery_img_list = glob(os.path.join(MONTGOMERY_IMG,'*.png'))
    montgomery_leftmask_list = glob(os.path.join(MONTGOMERY_LEFTMASK_IMG,'*.png'))
    montgomery_rightmask_list = glob(os.path.join(MONTGOMERY_RIGHTMASK_IMG,'*.png'))

    if not os.listdir(MONTGOMERY_COMBINEDMASK_IMG):
        for leftmaskpath, rightmaskpath in zip(montgomery_leftmask_list, montgomery_rightmask_list):
            lm_base = os.path.basename(leftmaskpath)
            rm_base = os.path.basename(rightmaskpath)
            if lm_base == rm_base:
                leftmask = cv2.imread(leftmaskpath, cv2.IMREAD_GRAYSCALE)
                rightmask = cv2.imread(rightmaskpath, cv2.IMREAD_GRAYSCALE)
        
                mask = np.maximum(leftmask, rightmask)
                cv2.imwrite(os.path.join(MONTGOMERY_COMBINEDMASK_IMG,lm_base),mask)

    montgomery_combinedmask_list = glob(os.path.join(MONTGOMERY_COMBINEDMASK_IMG,'*.png'))
    bn_2_imgpath = { os.path.basename(i):i for i in montgomery_img_list}
    bn_2_maskpath = { os.path.basename(i):i for i in montgomery_combinedmask_list }
    
    train_cvs_path = os.path.join(path,'train_image_mask.csv')
    eval_cvs_path = os.path.join(path,'eval_image_mask.csv')


    df = pd.DataFrame({'basename': list(bn_2_imgpath.keys()), 'imgpath': list(bn_2_imgpath.values())})
    df['mask'] = df.basename.map(bn_2_maskpath)
    df.drop(['basename'], axis=1, inplace=True)
    df_train = df[:100]
    df_val = df[100:]
    df_train.to_csv(train_cvs_path, index=False)
    df_val.to_csv(eval_cvs_path, index=False)


    return train_cvs_path, eval_cvs_path, path 

# path = load_and_set_data()