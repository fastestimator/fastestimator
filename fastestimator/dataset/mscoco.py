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
"""Download MS COCO 2017 dataset."""
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress
from pycocotools.coco import COCO

wget.callback_progress = callback_progress


def _download_data(link, data_path, idx, total_idx):
    if not os.path.exists(data_path):
        print("Downloading data to {}, file: {} / {}".format(data_path, idx + 1, total_idx))
        wget.download(link, data_path, bar=bar_custom)


def _extract_data(zip_path, data_path, idx, total_idx):
    if not os.path.exists(data_path):
        print("Extracting {}, file {} / {}".format(zip_path, idx + 1, total_idx))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(os.path.dirname(zip_path))


def _generate_object_data(path, image_name, image_id, data_temp, coco_gt_instance, mask_folder):
    anns_ids = coco_gt_instance.getAnnIds(imgIds=image_id, iscrowd=False)
    num_obj = len(anns_ids)
    if num_obj == 0:
        keep_data = False
    else:
        keep_data = True
        mask_file = os.path.join(mask_folder, image_name.replace("jpg", "png"))
        write_mask = not os.path.exists(mask_file)
        data_temp["x1"], data_temp["y1"], data_temp["width"], data_temp["height"], data_temp["obj_label"] = [], [], [], [], []
        data_temp["num_obj"] = num_obj
        anns = coco_gt_instance.loadAnns(anns_ids)
        for idx, ann in enumerate(anns):
            if write_mask:
                if idx == 0:
                    mask = coco_gt_instance.annToMask(ann=ann)
                else:
                    mask = np.clip(mask + (idx + 1) * coco_gt_instance.annToMask(ann=ann), None, idx + 1)
            data_temp["x1"].append(ann['bbox'][0])
            data_temp["y1"].append(ann['bbox'][1])
            data_temp["width"].append(ann['bbox'][2])
            data_temp["height"].append(ann['bbox'][3])
            data_temp["obj_label"].append(ann['category_id'])
        if write_mask:
            cv2.imwrite(mask_file, mask)
        data_temp["obj_mask"] = os.path.relpath(mask_file, path)
    return keep_data


def _generate_caption_data(image_id, data_temp, coco_gt_caption):
    anns_ids = coco_gt_caption.getAnnIds(imgIds=image_id)
    if not anns_ids:
        keep_data = False
    else:
        keep_data = True
        anns = coco_gt_caption.loadAnns(anns_ids)
        data_temp["caption"] = []
        for ann in anns:
            data_temp["caption"].append(ann['caption'])
    return keep_data


def _generate_data(data, path, image_folder, mask_folder, image_names, groundtruth):
    logging_interval = len(image_names) // 20
    for idx, image_name in enumerate(image_names):
        if idx % logging_interval == 0:
            print("Generating data from {}, progress: {:.1%}".format(image_folder, idx / len(image_names)))
        image_id = int(os.path.splitext(image_name)[0])
        keep_data = True
        data_temp = {"image": os.path.relpath(os.path.join(image_folder, image_name), path), "id": image_id}
        if "object" in groundtruth:
            keep_data = _generate_object_data(path, image_name, image_id, data_temp, groundtruth["object"], mask_folder)
        if "caption" in groundtruth and keep_data:
            keep_data = _generate_caption_data(image_id, data_temp, groundtruth["caption"])
        if keep_data:
            for key, value in data_temp.items():
                data[key].append(value)


def _generate_csv(path, load_object, load_caption, csv_file, image_folder, mask_folder, instance_gt, caption_gt):
    if not os.path.exists(csv_file):
        groundtruth = {}
        data = {'image': [], 'id': []}
        image_names = os.listdir(image_folder)
        if load_object:
            os.makedirs(mask_folder, exist_ok=True)
            groundtruth["object"] = COCO(instance_gt)
            data["num_obj"] = []
            data["x1"] = []
            data["y1"] = []
            data["width"] = []
            data["height"] = []
            data["obj_label"] = []
            data["obj_mask"] = []
        if load_caption:
            groundtruth["caption"] = COCO(caption_gt)
            data["caption"] = []
        _generate_data(data, path, image_folder, mask_folder, image_names, groundtruth)
        df = pd.DataFrame(data=data)
        df.to_csv(csv_file, index=False)


def _get_csv_name(base_name, load_object, load_caption, path):
    csv_name = base_name
    if load_object:
        csv_name += "_object"
    if load_caption:
        csv_name += "_caption"
    csv_name += ".csv"
    return os.path.join(path, csv_name)


def load_data(path=None, load_object=True, load_caption=False):
    """Download the COCO dataset to local storage, if not already downloaded. This will generate train and val
    csv files.

    Args:
        path (str, optional): The path to store the COCO data. When `path` is not provided, will save at
            `fastestimator_data` under home directory.
        load_object (bool, optional): whether to get object-related data, defaults to True.
        load_caption (bool, optional): whether to get caption-related data, defaults to False.

    Returns:
        train_csv (str): Path to the train summary csv file, containing the following columns:
            * image (str): image directory relative to the returned path
            * num_obj (int): number of objects within the image (available when object = True)
            * x1 (list): the top left x coordinate of object bounding boxes (available when object = True)
            * y1 (list): the top left y coordinate of object bounding boxes (available when object = True)
            * width (list): the width of object bounding boxes (available when object = True)
            * height (list): the height of object bounding boxes (available when object = True)
            * obj_label (list): categorical labels of each objects (available when object = True)
            * obj_mask (str): mask directory relative to the returned path, with pixel value being the object order
                             (available when object = True)
            * caption (list): list of captions for the image (available when caption = True)
        val_csv (str): Path to the eval summary csv file, the columns are the same as train_csv.
        path (str): Path to data directory.
    """
    if path is None:
        path = os.path.join(str(Path.home()), 'fastestimator_data', 'MSCOCO2017')
    else:
        path = os.path.join(os.path.abspath(path), 'MSCOCO2017')
    os.makedirs(path, exist_ok=True)
    #download the zip data
    zip_files = [("train2017.zip", 'http://images.cocodataset.org/zips/train2017.zip'),
                 ("val2017.zip", 'http://images.cocodataset.org/zips/val2017.zip'),
                 ("annotations_trainval2017.zip",
                  'http://images.cocodataset.org/annotations/annotations_trainval2017.zip')]
    for idx, (zip_file, zip_link) in enumerate(zip_files):
        _download_data(zip_link, os.path.join(path, zip_file), idx, len(zip_files))
    #extract data
    extract_folders = [("train2017", "train2017.zip"), ("val2017", "val2017.zip"),
                       ("annotations", "annotations_trainval2017.zip")]
    for idx, (folder, zip_file) in enumerate(extract_folders):
        _extract_data(os.path.join(path, zip_file), os.path.join(path, folder), idx, len(extract_folders))
    #generate csv
    train_csv = _get_csv_name("train", load_object, load_caption, path)
    val_csv = _get_csv_name("val", load_object, load_caption, path)
    _generate_csv(path,
                  load_object,
                  load_caption,
                  train_csv,
                  os.path.join(path, "train2017"),
                  os.path.join(path, "mask_train2017"),
                  os.path.join(path, "annotations", "instances_train2017.json"),
                  os.path.join(path, "annotations", "captions_train2017.json"))
    _generate_csv(path,
                  load_object,
                  load_caption,
                  val_csv,
                  os.path.join(path, "val2017"),
                  os.path.join(path, "mask_val2017"),
                  os.path.join(path, "annotations", "instances_val2017.json"),
                  os.path.join(path, "annotations", "captions_val2017.json"))
    return train_csv, val_csv, path
