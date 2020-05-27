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
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import wget
from pycocotools.coco import COCO

from fastestimator.dataset.dir_dataset import DirDataset
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import Suppressor
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


@traceable()
class MSCOCODataset(DirDataset):
    """A specialized DirDataset to handle MSCOCO data.

    This dataset combines images from the MSCOCO data directory with their corresponding bboxes, masks, and captions.

    Args:
        image_dir: The path the directory containing MSOCO images.
        annotation_file: The path to the file containing annotation data.
        caption_file: The path the file containing caption data.
        include_bboxes: Whether images should be paired with their associated bounding boxes. If true, images without
            bounding boxes will be ignored and other images may be oversampled in order to take their place.
        include_masks: Whether images should be paired with their associated masks. If true, images without masks will
            be ignored and other images may be oversampled in order to take their place.
        include_captions: Whether images should be paired with their associated captions. If true, images without
            captions will be ignored and other images may be oversampled in order to take their place.
        min_bbox_area: Bounding boxes with a total area less than `min_bbox_area` will be discarded.
    """

    instances: Optional[COCO]
    captions: Optional[COCO]

    def __init__(self,
                 image_dir: str,
                 annotation_file: str,
                 caption_file: str,
                 include_bboxes: bool = True,
                 include_masks: bool = False,
                 include_captions: bool = False,
                 min_bbox_area=1.0) -> None:
        super().__init__(root_dir=image_dir, data_key="image", recursive_search=False)
        if include_masks:
            assert include_bboxes, "must include bboxes with mask data"
        self.include_bboxes = include_bboxes
        self.include_masks = include_masks
        self.min_bbox_area = min_bbox_area
        with Suppressor():
            self.instances = COCO(annotation_file)
            self.captions = COCO(caption_file) if include_captions else None

    def __getitem__(self, index: Union[int, str]) -> Union[Dict[str, Any], np.ndarray, List[Any]]:
        """Look up data from the dataset.

        Args:
            index: Either an int corresponding to a particular element of data, or a string in which case the
                corresponding column of data will be returned. If bboxes, masks, or captions are required and the data
                at the desired index does not have one or more of these features, then data from a random index which
                does have all necessary features will be fetched instead.

        Returns:
            A data dictionary if the index was an int, otherwise a column of data in list format.
        """
        has_data = False
        response = {}
        while not has_data:
            has_box, has_mask, has_caption = True, True, True
            response = self._get_single_item(index)
            if isinstance(index, str):
                return response
            if self.include_bboxes and not response["bbox"]:
                has_box = False
            if self.include_masks and not response["mask"]:
                has_mask = False
            if self.captions and not response["caption"]:
                has_caption = False
            has_data = has_box and has_mask and has_caption
            index = np.random.randint(len(self))
        return response

    def _get_single_item(self, index: Union[int, str]) -> Union[Dict[str, Any], np.ndarray, List[Any]]:
        """Look up data from the dataset.

        Args:
            index: Either an int corresponding to a particular element of data, or a string in which case the
                corresponding column of data will be returned.

        Returns:
            A data dictionary if the index was an int, otherwise a column of data in list format.
        """
        response = super().__getitem__(index)
        if isinstance(index, str):
            return response
        else:
            response = deepcopy(response)
        image = response["image"]
        image_id = int(os.path.splitext(os.path.basename(image))[0])
        response["image_id"] = image_id
        if self.include_bboxes:
            self._populate_instance_data(response, image_id)
        if self.captions:
            self._populate_caption_data(response, image_id)
        return response

    def _populate_instance_data(self, data: Dict[str, Any], image_id: int) -> None:
        """Add instance data to a data dictionary.

        Args:
            data: The dictionary to be augmented.
            image_id: The id of the image for which to find data.
        """
        data["bbox"] = []
        if self.include_masks:
            data["mask"] = []
        annotation_ids = self.instances.getAnnIds(imgIds=image_id, iscrowd=False)
        if annotation_ids:
            annotations = self.instances.loadAnns(annotation_ids)
            for annotation in annotations:
                if annotation["bbox"][2] * annotation["bbox"][3] > self.min_bbox_area:
                    data["bbox"].append(tuple(annotation['bbox'] + [annotation['category_id']]))
                    if self.include_masks:
                        data["mask"].append(self.instances.annToMask(annotation))

    def _populate_caption_data(self, data: Dict[str, Any], image_id: int) -> None:
        """Add captions to a data dictionary.

        Args:
            data: The dictionary to be augmented.
            image_id: The id of the image for which to find captions.
        """
        data["caption"] = []
        annotation_ids = self.captions.getAnnIds(imgIds=image_id)
        if annotation_ids:
            annotations = self.captions.loadAnns(annotation_ids)
            for annotation in annotations:
                data["caption"].append(annotation['caption'])


def load_data(root_dir: Optional[str] = None,
              load_bboxes: bool = True,
              load_masks: bool = False,
              load_captions: bool = False) -> Tuple[MSCOCODataset, MSCOCODataset]:
    """Load and return the COCO dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        load_bboxes: Whether to load bbox-related data.
        load_masks: Whether to load mask data (in the form of an array of 1-hot images).
        load_captions: Whether to load caption-related data.

    Returns:
        (train_data, eval_data)
    """
    if root_dir is None:
        root_dir = os.path.join(str(Path.home()), 'fastestimator_data', 'MSCOCO2017')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'MSCOCO2017')
    os.makedirs(root_dir, exist_ok=True)

    train_data = os.path.join(root_dir, "train2017")
    eval_data = os.path.join(root_dir, "val2017")
    annotation_data = os.path.join(root_dir, "annotations")

    files = [(train_data, "train2017.zip", 'http://images.cocodataset.org/zips/train2017.zip'),
             (eval_data, "val2017.zip", 'http://images.cocodataset.org/zips/val2017.zip'),
             (annotation_data,
              "annotations_trainval2017.zip",
              'http://images.cocodataset.org/annotations/annotations_trainval2017.zip')]

    for data_dir, zip_name, download_url in files:
        if not os.path.exists(data_dir):
            zip_path = os.path.join(root_dir, zip_name)
            # Download
            if not os.path.exists(zip_path):
                print("Downloading {} to {}".format(zip_name, root_dir))
                wget.download(download_url, zip_path, bar=bar_custom)
            # Extract
            print("Extracting {}".format(zip_name))
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(os.path.dirname(zip_path))

    train_annotation = os.path.join(annotation_data, "instances_train2017.json")
    eval_annotation = os.path.join(annotation_data, "instances_val2017.json")
    train_captions = os.path.join(annotation_data, "captions_train2017.json")
    eval_captions = os.path.join(annotation_data, "captions_val2017.json")
    train_ds = MSCOCODataset(train_data,
                             train_annotation,
                             train_captions,
                             include_bboxes=load_bboxes,
                             include_masks=load_masks,
                             include_captions=load_captions)
    eval_ds = MSCOCODataset(eval_data,
                            eval_annotation,
                            eval_captions,
                            include_bboxes=load_bboxes,
                            include_masks=load_masks,
                            include_captions=load_captions)
    return train_ds, eval_ds
