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
"""COCO Mean average precisin (mAP) implementation."""
from collections import defaultdict
from typing import Dict

import numpy as np

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from pycocotools import mask as maskUtils


class MeanAveragePrecision(Trace):
    """Calculate COCO mean average precision.

    Args:
    Returns:
    """
    def __init__(self,
                 num_classes: int,
                 true_key='bbox',
                 pred_key: str = 'pred',
                 mode: str = "eval",
                 output_name=("mAP", "AP50", "AP75")) -> None:
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)

        #assert len(self.output_name) == 3, 'MeanAvgPrecision trace adds 3 fields mAP AP50 AP75 to state dict'

        self.iou_thres = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05).astype(np.int) + 1, endpoint=True)
        self.recall_thres = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01).astype(np.int) + 1, endpoint=True)
        self.categories = range(1, num_classes + 1)  # MSCOCO style class label starts from 1
        self.max_detection = 100
        self.image_ids = []

        # eval
        self.evalimgs = {}
        self.eval = {}
        self.ids_in_epoch = -1  # reset per epoch

        # reset per batch
        self.gt = defaultdict(list)  # gt for evaluation
        self.det = defaultdict(list)
        self.batch_image_ids = []  # img_ids per batch
        self.ious = defaultdict(list)
        self.ids_unique = []
        self.ids_batch_to_epoch = {}

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def _get_id_in_epoch(self, idx_in_batch):
        """Get unique image id in epoch.

        Args:
            idx_in_batch:

        Returns:
        """
        # for this batch
        num_unique_id_previous = len(np.unique(self.ids_unique))
        self.ids_unique.append(idx_in_batch)
        num_unique_id = len(np.unique(self.ids_unique))

        if num_unique_id > num_unique_id_previous:
            # for epoch
            self.ids_in_epoch += 1
            self.ids_batch_to_epoch[idx_in_batch] = self.ids_in_epoch
        return self.ids_in_epoch

    def on_epoch_begin(self, data: Data):
        """Reset"""
        self.image_ids = []  # append all the image ids coming from each iteration
        self.evalimgs = {}
        self.eval = {}
        self.ids_in_epoch = -1

    def on_batch_begin(self, data: Data):
        """Reset"""
        self.gt = defaultdict(list)  # gt for evaluation
        self.det = defaultdict(list)  # dt for evaluation
        self.batch_image_ids = []  # img_ids per batch

        self.ious = defaultdict(list)
        self.ids_unique = []
        self.ids_batch_to_epoch = {}

    def on_batch_end(self, data: Data):
        #########read det, gt

        pred = list(map(to_number, data[self.pred_key]))  # pred is list (batch, ) of np.ndarray (?, 6)
        gt = to_number(data[self.true_key])  # gt is np.array (batch, box, 5), box dimension is padded

        ground_truth_bb = []
        for gt_item in gt:
            idx_in_batch, x1, y1, w, h, label = gt_item
            label = int(label)
            id_epoch = self._get_id_in_epoch(idx_in_batch)
            self.batch_image_ids.append(id_epoch)
            self.image_ids.append(id_epoch)
            tmp_dict = {'idx': id_epoch, 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'label': label}
            ground_truth_bb.append(tmp_dict)

        predicted_bb = []
        for pred_item in pred:
            idx_in_batch, x1, y1, w, h, label, score = pred_item
            label = int(label)
            id_epoch = self.ids_batch_to_epoch[idx_in_batch]
            self.image_ids.append(id_epoch)
            tmp_dict = {'idx': id_epoch, 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'label': label, 'score': score}
            predicted_bb.append(tmp_dict)

        for dict_elem in ground_truth_bb:
            self.gt[dict_elem['idx'], dict_elem['label']].append(dict_elem)

        for dict_elem in predicted_bb:
            self.det[dict_elem['idx'], dict_elem['label']].append(dict_elem)
        #########end of read det, gt

        # compute iou
        self.ious = {(img_id, cat_id): self.compute_iou(self.det[img_id, cat_id], self.gt[img_id, cat_id])
                     for img_id in self.batch_image_ids for cat_id in self.categories}

        for cat_id in self.categories:
            for img_id in self.batch_image_ids:
                self.evalimgs[(cat_id, img_id)] = self.evaluate_img(cat_id, img_id)

    def on_epoch_end(self, data: Data):
        self.accumulate()

        mean_ap = self.summarize()
        ap50 = self.summarize(iou=0.5)
        ap75 = self.summarize(iou=0.75)

        data[self.output_name[0]] = mean_ap
        data[self.output_name[1]] = ap50
        data[self.output_name[2]] = ap75

    def accumulate(self):
        """Generate precision recall curve"""
        key_list = self.evalimgs
        key_list = sorted(key_list)
        eval_list = [self.evalimgs[key] for key in key_list]

        self.image_ids = np.unique(self.image_ids)

        num_iou_thresh = len(self.iou_thres)
        num_recall_thresh = len(self.recall_thres)
        num_categories = len(self.categories)
        cat_list_zeroidx = [n for n, cat in enumerate(self.categories)]

        num_imgs = len(self.image_ids)
        maxdets = self.max_detection

        precision_marix = -np.ones((num_iou_thresh, num_recall_thresh, num_categories))
        recall = -np.ones((num_iou_thresh, num_categories))
        scores_matrix = -np.ones((num_iou_thresh, num_recall_thresh, num_categories))

        # loop through category
        for cat_index in cat_list_zeroidx:
            Nk = cat_index * num_imgs
            # each element is one image inside this category
            eval_by_category = [eval_list[Nk + img_idx] for img_idx in range(num_imgs)]
            eval_by_category = [e for e in eval_by_category if not e is None]

            # no image inside this category
            if len(eval_by_category) == 0:
                continue

            det_scores = np.concatenate([e['dtScores'][0:maxdets] for e in eval_by_category])

            # sort from high score to low score, is this necessary?
            sorted_score_inds = np.argsort(-det_scores, kind='mergesort')

            det_scores_sorted = det_scores[sorted_score_inds]
            det_matrix = np.concatenate([e['dtMatches'][:, 0:maxdets] for e in eval_by_category],
                                        axis=1)[:, sorted_score_inds]

            # number of all image gts in one category
            num_all_gt = np.sum([e['num_gt'] for e in eval_by_category])
            # for all images no gt inside this category
            if num_all_gt == 0:
                continue

            tps = det_matrix > 0
            fps = det_matrix == 0

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for t, (true_positives, false_positives) in enumerate(zip(tp_sum, fp_sum)):
                true_positives = np.array(true_positives)
                false_positives = np.array(false_positives)
                num_tps = len(true_positives)
                recall = true_positives / num_all_gt
                precision = true_positives / (false_positives + true_positives + np.spacing(1))

                q = np.zeros((num_recall_thresh, ))
                score = np.zeros((num_recall_thresh, ))

                if num_tps:
                    recall[t, cat_index] = recall[-1]
                else:
                    recall[t, cat_index] = 0

                precision = precision.tolist()
                q = q.tolist()

                for i in range(num_tps - 1, 0, -1):
                    if precision[i] > precision[i - 1]:
                        precision[i - 1] = precision[i]

                sorted_score_inds = np.searchsorted(recall, self.recall_thres, side='left')

                try:
                    for recall_index, precision_index in enumerate(sorted_score_inds):
                        q[recall_index] = precision[precision_index]
                        score[recall_index] = det_scores_sorted[precision_index]
                except:
                    pass

                precision_marix[t, :, cat_index] = np.array(q)
                scores_matrix[t, :, cat_index] = np.array(score)

        self.eval = {
            'counts': [num_iou_thresh, num_recall_thresh, num_categories],
            'precision': precision_marix,
            'recall': recall,
            'scores': scores_matrix,
        }

    def summarize(self, iou=None):
        s = self.eval['precision']
        if iou is not None:
            t = np.where(iou == self.iou_thres)[0]
            s = s[t]

        s = s[:, :, :]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s

    def evaluate_img(self, cat_id: int, img_id: int) -> Dict:
        """Evaluate one image one category.

        Args:
            cat_id:
            img_id:

        Returns:

        """
        det = self.det[img_id, cat_id]
        gt = self.gt[img_id, cat_id]
        num_dt = len(det)
        num_gt = len(gt)
        if num_gt == 0 and num_dt == 0:
            return None

        # sort detections, is ths necessary?
        det_index = np.argsort([-d['score'] for d in det], kind='mergesort')

        # cap to max_detection
        det = [det[i] for i in det_index[0:self.max_detection]]

        # put iou into category matrix
        iou_mat = self.ious[img_id, cat_id]

        num_iou_thresh = len(self.iou_thres)

        det_matrix = np.zeros((num_iou_thresh, num_dt))
        gt_matrix = np.zeros((num_iou_thresh, num_gt))

        if len(iou_mat) != 0:
            # loop through each iou thresh
            for thres_idx, thres_value in enumerate(self.iou_thres):
                # loop through each detection
                for det_idx, det_elem in enumerate(det):
                    m = -1
                    iou = min([thres_value, 1 - 1e-10])

                    for gt_idx, gt_elem in enumerate(gt):
                        if gt_matrix[thres_idx, gt_idx] > 0:
                            continue
                        if iou_mat[det_idx, gt_idx] >= iou:
                            iou = iou_mat[det_idx, gt_idx]
                            m = gt_idx

                    if m != -1:
                        det_matrix[thres_idx, det_idx] = gt[m]['idx']
                        gt_matrix[thres_idx, m] = 1

        return {
            'image_id': img_id,
            'category_id': cat_id,
            'gtIds': [g['idx'] for g in gt],
            'dtMatches': det_matrix,
            'gtMatches': gt_matrix,
            'dtScores': [d['score'] for d in det],
            'num_gt': num_gt,
        }

    def compute_iou(self, det: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Compute intersection over union.

        Args:
            det:
            gt:

        Returns:
            Intersection of union matrix.
        """
        num_dt = len(det)
        num_gt = len(gt)

        if num_gt == 0 and num_dt == 0:
            return []

        boxes_a = np.zeros(shape=(0, 4), dtype=float)
        boxes_b = np.zeros(shape=(0, 4), dtype=float)

        inds = np.argsort([-d['score'] for d in det], kind='mergesort')
        det = [det[i] for i in inds]
        if len(det) > self.max_detection:
            det = det[0:self.max_detection]

        boxes_a = [[dt_elem['x1'], dt_elem['y1'], dt_elem['w'], dt_elem['h']] for dt_elem in det]
        boxes_b = [[gt_elem['x1'], gt_elem['y1'], gt_elem['w'], gt_elem['h']] for gt_elem in gt]

        iscrowd = [0] * num_gt  # to leverage maskUtils.iou
        iou_dt_gt = maskUtils.iou(boxes_a, boxes_b, iscrowd)
        return iou_dt_gt
