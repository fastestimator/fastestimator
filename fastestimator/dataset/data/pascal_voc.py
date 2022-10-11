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
import os
import tarfile
import collections
from defusedxml.ElementTree import parse as ET_parse

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import wget

from fastestimator.dataset.dataset import FEDataset
from fastestimator.util.traceability_util import traceable
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress

VOC_COLORMAP = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@traceable()
class PascalVoc(FEDataset):
    """A specialized DirDataset to handle MSCOCO data.

    This dataset combines images from the MSCOCO data directory with their corresponding bboxes, masks, and captions.

    Args:
        image_dir: The path the directory containing MSOCO images.
        include_bboxes: Whether images should be paired with their associated bounding boxes.
        include_masks: Whether images should be paired with their associated masks.
        include_captions: Whether images should be paired with their associated captions.
        include_keypoints: Whether images should be paired with keypoints.
        min_bbox_area: Bounding boxes with a total area less than `min_bbox_area` will be discarded.
        replacement: If true, images without requested attributes will be ignored and other images may be oversampled in
            order to take their place.
    """
    def __init__(self,
                 root_dir: str,
                 include_bboxes: bool = False,
                 include_masks: bool = True,
                 min_bbox_area: float = 1.0,
                 image_set: str = "train") -> None:
        self.include_bboxes = include_bboxes
        self.include_masks = include_masks

        # TO-DO
        # Pick only those Bounding boxes with area greater than the "min_bbox_area"
        self.min_bbox_area = min_bbox_area

        valid_image_sets = ["train", "trainval", "val"]

        if image_set not in valid_image_sets:
            raise ValueError('image_set must be either "train", "trainval" or "val" but received : ' + image_set)

        if self.include_masks:
            splits_dir = os.path.join(root_dir, "ImageSets", "Segmentation")
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f)) as f:
                self.seg_file_names = [x.strip() for x in f.readlines()]

            image_dir = os.path.join(root_dir, "JPEGImages")
            self.seg_images = {x: os.path.join(image_dir, x + ".jpg") for x in self.seg_file_names}

            target_dir = os.path.join(root_dir, "SegmentationClass")
            self.seg_targets = {x: os.path.join(target_dir, x + ".png") for x in self.seg_file_names}
        else:
            self.seg_file_names = []

        if self.include_bboxes:
            splits_dir = os.path.join(root_dir, "ImageSets", "Main")
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f)) as f:
                self.bbox_file_names = [x.strip() for x in f.readlines()]

            image_dir = os.path.join(root_dir, "JPEGImages")
            self.box_images = {x: os.path.join(image_dir, x + ".jpg") for x in self.bbox_file_names}

            target_dir = os.path.join(root_dir, "Annotations")
            self.box_targets = {x: os.path.join(target_dir, x + ".xml") for x in self.bbox_file_names}

            self.cls2ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

        else:
            self.bbox_file_names = []

        # Merge filenames
        self.filenames = list(set(self.bbox_file_names).union(set(self.seg_file_names)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: Union[int, str]) -> Union[Dict[str, Any], np.ndarray, List[Any]]:
        """Fetch a data instance at a specified index.

        Args:
            index: Which datapoint to retrieve.
        Returns:
            The data dictionary from the specified index.
        """
        filename = self.filenames[index]
        ifbox = False
        ifseg = False
        bbox = []
        mask = []
        if self.include_bboxes:
            box_img, bbox, ifbox = self._populate_bbox_data(filename)
        if self.include_masks:
            seg_img, mask, ifseg = self._populate_mask_data(filename)

        if ifbox:
            image = box_img
        elif ifseg:
            image = seg_img
        else:
            image = []
        return {'image': image, 'mask': mask, 'bbox': bbox}

    def _populate_bbox_data(self, filename: str) -> None:
        """Add Bounding boxes data to a data dictionary.

        Args:
            filename: The filename of the image for which to find data.
        """
        if self.box_images.get(filename) == None:
            return None, [], False

        image = cv2.imread(self.box_images[filename])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.parse_voc_xml(ET_parse(self.box_targets[filename]).getroot())

        # TO-DO
        # The following function can also be used to output pose for each bbox
        bbox = self.get_objects(target)

        return image, bbox, True

    @staticmethod
    def parse_voc_xml(node) -> Dict[str, Any]:
        '''
        Parse xml file to read Bounding Box annotations of individual image files
        '''
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(PascalVoc.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def get_objects(self, ann_dict):
        '''
        Fetch and rearange Bounding box coordinates
        '''

        objects = ann_dict['annotation']['object']
        bbox = []
        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']
            cid = self.cls2ind[obj['name']]
            bbox.append([xmin, ymin, xmax, ymax, cid])

        return bbox

    def _populate_mask_data(self, filename: str) -> None:
        """Add Semantic Mask data to a data dictionary.

        Args:
            filename: The filename of the image for which to find data.
        """
        if self.seg_images.get(filename) == None:
            return None, [], False

        image = cv2.imread(self.seg_images[filename])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.seg_targets[filename])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # convert pixel masks to multidimentional
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)

        return image, segmentation_mask, True


def load_data(root_dir: Optional[str] = None, load_bboxes: bool = False,
              load_masks: bool = True) -> Tuple[PascalVoc, PascalVoc]:
    """Load and return the COCO dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        load_bboxes: Whether to load bbox-related data, in [x1, y1, w, h] format.
        load_masks: Whether to load mask data (in the form of an array of 1-hot images).


    Returns:
        (train_data, eval_data)
    """
    if root_dir is None:
        root_dir = os.path.join(str(Path.home()), 'fastestimator_data', 'Pascal_Voc')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'PascalVoc')
    os.makedirs(root_dir, exist_ok=True)

    data_folder = os.path.join(root_dir, "VOCdevkit", "VOC2012")

    files = [(data_folder,
              "VOCtrainval_11-May-2012.tar",
              'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')]

    for data_dir, zip_name, download_url in files:
        if not os.path.exists(data_dir):
            zip_path = os.path.join(root_dir, zip_name)
            # Download
            if not os.path.exists(zip_path):
                print("Downloading {} to {}".format(zip_name, root_dir))
                wget.download(download_url, zip_path, bar=bar_custom)
            # Extract
            print("Extracting {}".format(zip_name))
            with tarfile.open(zip_path) as tar_file:
                tar_file.extractall(os.path.dirname(zip_path))
    train_ds = PascalVoc(data_folder, include_bboxes=load_bboxes, include_masks=load_masks, image_set="train")
    eval_ds = PascalVoc(data_folder, include_bboxes=load_bboxes, include_masks=load_masks, image_set='val')
    return train_ds, eval_ds
