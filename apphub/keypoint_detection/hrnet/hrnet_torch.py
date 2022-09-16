# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import center_of_mass

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.dataset.data.mscoco import load_data
from fastestimator.op.numpyop.multivariate import LongestMaxSize, PadIfNeeded, Resize
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp, RemoveIf
from fastestimator.op.numpyop.univariate import ChannelTranspose, ReadImage
from fastestimator.op.tensorop.loss import MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.normalize import Normalize
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.trace import Trace


class HRnet(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    ResBlock(64, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64))
        self.stage2 = StageModule(num_module=1, num_block=(4, 4), c_in=(256, ), c_out=(32, 64))
        self.stage3 = StageModule(num_module=4, num_block=(4, 4, 4), c_in=(32, 64), c_out=(32, 64, 128))
        self.stage4 = StageModule(num_module=3, num_block=(4, 4, 4, 4), c_in=(32, 64, 128), c_out=(32, 64, 128, 256))
        self.final_layer = nn.Conv2d(32, num_classes, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2([x])  # starting from the second stage, x becomes a list
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_layer(x[0])  # for keypoint detection, only use the hightest resolution output
        x = torch.sigmoid(x)
        return x


class StageModule(nn.Module):
    def __init__(self, num_module, num_block, c_in, c_out):
        super().__init__()
        self.transition_module = self.make_transition(c_in, c_out)
        self.stage_module = nn.Sequential(*[HRModule(num_block, c_out) for _ in range(num_module)])

    def make_transition(self, c_in, c_out):
        num_branch_in, num_branch_out = len(c_in), len(c_out)
        transition_layers = []
        for idx in range(num_branch_out):
            if idx < num_branch_in:  # extending existing scale horizontally
                if c_in[idx] != c_out[idx]:
                    transition_layers.append(
                        nn.Sequential(nn.Conv2d(c_in[idx], c_out[idx], kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(c_out[idx]),
                                      nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:  # new vertical scale
                convs = []
                for j in range(idx + 1 - num_branch_in):
                    filter_in = c_in[-1]
                    filter_out = c_out[idx] if j == idx - num_branch_in else filter_in
                    convs.append(
                        nn.Sequential(nn.Conv2d(filter_in, filter_out, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(filter_out),
                                      nn.ReLU()))
                transition_layers.append(nn.Sequential(*convs))
        return nn.ModuleList(transition_layers)

    def forward(self, x):
        x = x + [x[-1] for _ in range(len(self.transition_module) - len(x))]  # make sure the x is padded with x[-1]
        x = [m(data) if m is not None else data for m, data in zip(self.transition_module, x)]
        x = self.stage_module(x)
        return x


class HRModule(nn.Module):
    def __init__(self, num_conv, c_out):
        super().__init__()
        self.branches = nn.ModuleList([self._make_branch(nconv, plane) for nconv, plane in zip(num_conv, c_out)])
        self.fuse_layers = self._make_fuse_layers(c_out)
        self.relu = nn.ReLU()

    def _make_branch(self, conv_count, plane):
        return nn.Sequential(*[BasicBlock(plane, plane) for _ in range(conv_count)])

    def _make_fuse_layers(self, channels):
        fuse_layers = []
        # length of c_out means number of branches
        for idx_out, planes_out in enumerate(channels):
            fuse_layer = []
            for idx_in, planes_in in enumerate(channels):
                if idx_in == idx_out:
                    fuse_layer.append(None)
                elif idx_in > idx_out:  # low-res -> high-res, need upsampling
                    fuse_layer.append(
                        nn.Sequential(nn.Conv2d(planes_in, planes_out, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(planes_out),
                                      nn.Upsample(scale_factor=2**(idx_in - idx_out), mode='nearest')))
                else:  # high-res -> low-res, need strided conv
                    convs = [
                        nn.Sequential(nn.Conv2d(planes_in, planes_in, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(planes_in),
                                      nn.ReLU()) for _ in range(idx_out - idx_in - 1)
                    ] + [
                        nn.Sequential(nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(planes_out))
                    ]
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        x = [branch(x_i) for branch, x_i in zip(self.branches, x)]
        x = [
            sum([x_i if layer is None else layer(x_i) for layer, x_i in zip(fuse_layers, x)])
            for fuse_layers in self.fuse_layers
        ]
        x = [self.relu(x_i) for x_i in x]
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        # when shape mismatch between input and conv output, apply input_conv to input
        if inplanes != planes * 4 or stride != 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        else:
            self.input_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.input_conv is not None:
            residual = self.input_conv(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        # when shape mismatch between input and conv output, apply input_conv to input
        if inplanes != planes or stride != 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.input_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.input_conv is not None:
            residual = self.input_conv(x)
        out += residual
        out = self.relu(out)
        return out


def collect_single_keypoint_ds(ds, cache_limit=None):
    images, keypoints, keypoint_bboxes = [], [], []
    for idx, item in enumerate(ds):
        for keypoint, keypoint_bbox in zip(item['keypoint'], item['keypoint_bbox']):
            images.append(item['image'])
            keypoints.append(keypoint)
            keypoint_bboxes.append(keypoint_bbox)
        if idx % 1000 == 0 or idx + 1 == len(ds):
            print("Loading data --- {} / {}".format(idx + 1, len(ds)))
        if cache_limit and idx + 1 == cache_limit:
            break
    return NumpyDataset(data={"image": images, "keypoint": keypoints, "keypoint_bbox": keypoint_bboxes})


class KeypointMask(NumpyOp):
    def forward(self, data, state):
        image, keypoint, kpscore = data
        height, width, _ = image.shape
        kp_mask = np.stack([self.gaussian(kp, ks, height, width) for kp, ks in zip(keypoint, kpscore)], axis=-1)
        return kp_mask

    def gaussian(self, kp, ks, height, width):
        x0, y0 = kp
        x0, y0 = int(x0), int(y0)
        mask = np.zeros((height, width), dtype="float32")
        if ks >= 1:
            y_min, y_max = max(y0 - 2, 0), min(y0 + 3, height)
            x_min, x_max = max(x0 - 2, 0), min(x0 + 3, width)
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    # only worry about the 5x5 around keypoint center
                    mask[yi, xi] = np.exp(-((xi - x0)**2 + (yi - y0)**2) / (2 * 1**2))
        return mask


class CropImageKeypoint(NumpyOp):
    def forward(self, data, state):
        image, keypoint_bbox, keypoint = data
        image = self._crop_image(image, keypoint_bbox)
        keypoints, kpscore = self._crop_keypoint(keypoint, keypoint_bbox)
        return image, keypoints, kpscore

    def _crop_image(self, image, bbox):
        x1, y1, box_w, box_h = bbox
        x2, y2 = x1 + box_w, y1 + box_h
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        image = image[y1:y2, x1:x2]
        return image

    def _crop_keypoint(self, keypoints, bbox):
        x1, y1, w, h = bbox
        kpscore = keypoints[:, -1]
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        kp_x = np.clip(keypoints[:, 0] - x1, a_min=0, a_max=w - 1)
        kp_y = np.clip(keypoints[:, 1] - y1, a_min=0, a_max=h - 1)
        keypoints = [(x, y) for x, y in zip(kp_x, kp_y)]
        return keypoints, kpscore


class KeypointAccuracy(Trace):
    def on_epoch_begin(self, data):
        self.TN, self.TP, self.FP, self.FN = 0, 0, 0, 0

    def on_batch_end(self, data):
        pred_mask = data["pred_mask"].numpy()
        gt_mask = data["kp_mask"].numpy()
        for pred, gt in zip(pred_mask, gt_mask):
            self.update_counts(pred, gt)

    def update_counts(self, pred, gt):
        num_channel = gt.shape[0]
        for idx in range(num_channel):
            pred_s, gt_s = pred[idx], gt[idx]
            gt_center = np.array(center_of_mass(gt_s)) if gt_s.max() == 1 else None
            if gt_center is None:
                if pred_s.max() >= 0.5:
                    self.FP += 1
                else:
                    self.TN += 1
            elif pred_s.max() < 0.5:
                # if no positive prediction and gt exists, then add 1 to false negative
                self.FN += 1
            else:
                pred_center = (np.median(np.where(pred_s == pred_s.max())[0]),
                               np.median(np.where(pred_s == pred_s.max())[1]))
                if np.linalg.norm(np.array(pred_center) - gt_center) > 3:
                    self.FP += 1
                else:
                    self.TP += 1

    def on_epoch_end(self, data):
        data.write_with_log("kp_acc", (self.TP + self.TN) / (self.TN + self.TP + self.FN + self.FP))
        data.write_with_log("FP", self.FP)
        data.write_with_log("TP", self.TP)
        data.write_with_log("FN", self.FN)
        data.write_with_log("TN", self.TN)


def get_estimator(data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  epochs=80,
                  batch_size=128,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  cache_limit=None):
    train_ds, eval_ds = load_data(root_dir=data_dir, load_bboxes=False, load_keypoints=True, replacement=False)
    train_ds = collect_single_keypoint_ds(train_ds, cache_limit=cache_limit)
    eval_ds = collect_single_keypoint_ds(eval_ds, cache_limit=cache_limit)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            CropImageKeypoint(inputs=("image", "keypoint_bbox", "keypoint"), outputs=("image", "keypoint", "kpscore")),
            LongestMaxSize(max_size=256, image_in="image", keypoints_in="keypoint", keypoint_params='xy'),
            PadIfNeeded(min_height=256,
                        min_width=256,
                        image_in="image",
                        keypoints_in="keypoint",
                        keypoint_params='xy',
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
            Resize(height=64,
                   width=64,
                   image_in="image",
                   image_out="image_down",
                   keypoints_in="keypoint",
                   keypoint_params='xy'),
            RemoveIf(fn=lambda x: len(x) < 17, replacement=False, inputs="keypoint"),
            KeypointMask(inputs=("image_down", "keypoint", "kpscore"), outputs="kp_mask"),
            ChannelTranspose(inputs=("image", "kp_mask"), outputs=("image", "kp_mask")),
            Delete(keys=("keypoint", "keypoint_bbox", "kpscore", "image_down"))
        ])
    model = fe.build(model_fn=HRnet, optimizer_fn="adam")
    network = fe.Network(ops=[
        Normalize(inputs="image", outputs="image", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ModelOp(inputs="image", model=model, outputs="pred_mask"),
        MeanSquaredError(inputs=("pred_mask", "kp_mask"), outputs="mse"),
        UpdateOp(model=model, loss_name="mse")
    ])
    traces = [
        KeypointAccuracy(inputs=("pred_mask", "kp_mask"), outputs=("kp_acc", "FP", "FN", "TP", "TN"), mode="eval"),
        BestModelSaver(model=model, save_dir=model_dir, metric="kp_acc", save_best_mode="max"),
        LRScheduler(model=model,
                    lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=epochs, init_lr=1e-3, min_lr=1e-4))
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
