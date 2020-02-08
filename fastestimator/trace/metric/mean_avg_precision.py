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
from collections import defaultdict

import numpy as np

from fastestimator.architecture.retinanet import _get_fpn_anchor_box
from fastestimator.trace import Trace
from pycocotools import mask as maskUtils


class MeanAvgPrecision(Trace):
    """Calculates mean avg precision for various ios. Based out of cocoapi
    """
    def __init__(self, num_classes, input_shape, pred_key, gt_key, mode="eval", output_name=("mAP", "AP50", "AP75")):
        super().__init__(outputs=output_name, mode=mode)
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.output_name = output_name
        assert len(self.output_name) == 3, 'MeanAvgPrecision  trace adds  3  fields mAP AP50 AP75 to state '

        self.iou_thres = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05).astype(np.int) + 1, endpoint=True)
        self.rec_thres = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01).astype(np.int) + 1, endpoint=True)
        self.categories = [n + 1 for n in range(num_classes)]  # MSCOCO style class label starts from 1
        self.maxdets = 100
        self.image_ids = []
        self.anch_box = _get_fpn_anchor_box(input_shape=input_shape)[0]

    def get_ids_in_epoch(self, idx_in_batch):
        unique_cnt_pr = len(np.unique(self.ids_unique))
        self.ids_unique.append(idx_in_batch)
        unique_cnt_ltr = len(np.unique(self.ids_unique))
        if unique_cnt_ltr > unique_cnt_pr:
            self.ids_in_epoch += 1
            self.ids_batch_to_epoch[idx_in_batch] = self.ids_in_epoch
        return self.ids_in_epoch

    def on_epoch_begin(self, state):
        self.image_ids = []  # append all the image ids coming from each iteration
        self.evalimgs = {}
        self.eval = {}
        self.ids_in_epoch = -1

    def on_batch_begin(self, state):
        self.gt = defaultdict(list)  # gt for evaluation
        self.dt = defaultdict(list)  # dt for evaluation
        self.batch_image_ids = []  # img_ids per batch
        self.ious = defaultdict(list)
        self.ids_unique = []
        self.ids_batch_to_epoch = {}

    def on_batch_end(self, state):

        pred = state["batch"][self.pred_key]
        pred = pred.numpy()
        gt = state["batch"][self.gt_key]
        gt = gt.numpy()

        ground_truth_bb = []
        for gt_item in gt:
            idx_in_batch, x1, y1, w, h, cls = gt_item
            idx_in_batch, cls = int(idx_in_batch), int(cls)
            id_epoch = self.get_ids_in_epoch(idx_in_batch)
            self.batch_image_ids.append(id_epoch)
            self.image_ids.append(id_epoch)
            tmp_dict = {'idx': id_epoch, 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'cls': cls}
            ground_truth_bb.append(tmp_dict)

        predicted_bb = []
        for pred_item in pred:
            idx_in_batch, x1, y1, w, h, cls, score = pred_item
            idx_in_batch, cls = int(idx_in_batch), int(cls)
            id_epoch = self.ids_batch_to_epoch[idx_in_batch]
            self.image_ids.append(id_epoch)
            tmp_dict = {'idx': id_epoch, 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'cls': cls, 'score': score}
            predicted_bb.append(tmp_dict)

        for dict_elem in ground_truth_bb:
            self.gt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)
        for dict_elem in predicted_bb:
            self.dt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)

        self.ious = {(img_id, cat_id): self.compute_iou(self.dt[img_id, cat_id], self.gt[img_id, cat_id])
                     for img_id in self.batch_image_ids for cat_id in self.categories}
        for cat_id in self.categories:
            for img_id in self.batch_image_ids:
                self.evalimgs[(cat_id, img_id)] = self.evaluate_img(cat_id, img_id)

    def on_epoch_end(self, state):
        self.accumulate()
        mean_ap = self.summarize()
        ap50 = self.summarize(iou=0.5)
        ap75 = self.summarize(iou=0.75)
        state[self.output_name[0]] = mean_ap
        state[self.output_name[1]] = ap50
        state[self.output_name[2]] = ap75

    def accumulate(self):
        key_list = self.evalimgs
        key_list = sorted(key_list)
        eval_list = [self.evalimgs[key] for key in key_list]

        self.image_ids = np.unique(self.image_ids)

        T = len(self.iou_thres)
        R = len(self.rec_thres)
        K = len(self.categories)
        cat_list_zeroidx = [n for n, cat in enumerate(self.categories)]

        I = len(self.image_ids)
        maxdets = self.maxdets

        precision = -np.ones((T, R, K))
        recall = -np.ones((T, K))
        scores = -np.ones((T, R, K))

        for k in cat_list_zeroidx:
            Nk = k * I
            E = [eval_list[Nk + img_idx] for img_idx in range(I)]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dt_scores = np.concatenate([e['dtScores'][0:maxdets] for e in E])
            inds = np.argsort(-dt_scores, kind='mergesort')
            dt_scores_sorted = dt_scores[inds]
            dtm = np.concatenate([e['dtMatches'][:, 0:maxdets] for e in E], axis=1)[:, inds]

            npig = np.sum([e['num_gt'] for e in E])
            if npig == 0:
                continue

            tps = dtm > 0
            fps = dtm == 0

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                q = np.zeros((R, ))
                ss = np.zeros((R, ))

                if nd:
                    recall[t, k] = rc[-1]
                else:
                    recall[t, k] = 0
                pr = pr.tolist()
                q = q.tolist()

                for i in range(nd - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]
                inds = np.searchsorted(rc, self.rec_thres, side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        ss[ri] = dt_scores_sorted[pi]
                except:
                    pass
                precision[t, :, k] = np.array(q)
                scores[t, :, k] = np.array(ss)
        self.eval = {
            'counts': [T, R, K],
            'precision': precision,
            'recall': recall,
            'scores': scores, }

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

    def evaluate_img(self, cat_id, img_id):
        dt = self.dt[img_id, cat_id]
        gt = self.gt[img_id, cat_id]
        num_dt = len(dt)
        num_gt = len(gt)
        if num_gt == 0 and num_dt == 0:
            return None

        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:self.maxdets]]

        iou_mat = self.ious[img_id, cat_id]
        T = len(self.iou_thres)

        dtm = np.zeros((T, num_dt))
        gtm = np.zeros((T, num_gt))

        if len(iou_mat) != 0:
            for thres_idx, thres_elem in enumerate(self.iou_thres):
                for dt_idx, dt_elem in enumerate(dt):
                    m = -1
                    iou = min([thres_elem, 1 - 1e-10])
                    for gt_idx, gt_elem in enumerate(gt):
                        if gtm[thres_idx, gt_idx] > 0:
                            continue
                        if iou_mat[dt_idx, gt_idx] >= iou:
                            iou = iou_mat[dt_idx, gt_idx]
                            m = gt_idx

                    if m != -1:
                        dtm[thres_idx, dt_idx] = gt[m]['idx']
                        gtm[thres_idx, m] = 1

        return {
            'image_id': img_id,
            'category_id': cat_id,
            'gtIds': [g['idx'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'num_gt': num_gt,
        }

    def compute_iou(self, dt, gt):
        num_dt = len(dt)
        num_gt = len(gt)

        if num_gt == 0 and num_dt == 0:
            return []

        boxes_a = np.zeros(shape=(0, 4), dtype=float)
        boxes_b = np.zeros(shape=(0, 4), dtype=float)

        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxdets:
            dt = dt[0:self.maxdets]

        boxes_a = [[dt_elem['x1'], dt_elem['y1'], dt_elem['w'], dt_elem['h']] for dt_elem in dt]
        boxes_b = [[gt_elem['x1'], gt_elem['y1'], gt_elem['w'], gt_elem['h']] for gt_elem in gt]

        iscrowd = [0 for o in gt]  # to leverage maskUtils.iou
        iou_dt_gt = maskUtils.iou(boxes_a, boxes_b, iscrowd)
        return iou_dt_gt
