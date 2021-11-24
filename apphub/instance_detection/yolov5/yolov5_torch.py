# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import math
import random
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from albumentations import BboxParams
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import CenterCrop, HorizontalFlip, LongestMaxSize, PadIfNeeded
from fastestimator.op.numpyop.univariate import ReadImage, ToArray
from fastestimator.op.tensorop import Average, TensorOp
from fastestimator.op.tensorop.loss import LossOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import MeanAveragePrecision
from fastestimator.util import get_num_devices


# This dataset selects 4 images and their bboxes
class PreMosaicDataset(Dataset):
    def __init__(self, mscoco_ds):
        self.mscoco_ds = mscoco_ds

    def __len__(self):
        return len(self.mscoco_ds)

    def __getitem__(self, idx):
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        samples = [self.mscoco_ds[i] for i in indices]
        return {
            "image1": samples[0]["image"],
            "bbox1": samples[0]["bbox"],
            "image2": samples[1]["image"],
            "bbox2": samples[1]["bbox"],
            "image3": samples[2]["image"],
            "bbox3": samples[2]["bbox"],
            "image4": samples[3]["image"],
            "bbox4": samples[3]["bbox"]
        }


class CombineMosaic(NumpyOp):
    def forward(self, data, state):
        image1, image2, image3, image4, bbox1, bbox2, bbox3, bbox4 = data
        images = [image1, image2, image3, image4]
        bboxes = [bbox1, bbox2, bbox3, bbox4]
        images_new, boxes_new = self._combine_images_boxes(images, bboxes)
        return images_new, boxes_new

    def _combine_images_boxes(self, images, bboxes):
        s = 640
        yc, xc = int(random.uniform(320, 960)), int(random.uniform(320, 960))
        images_new = np.full((1280, 1280, 3), fill_value=114, dtype=np.uint8)
        bboxes_new = []
        for idx, (image, bbox) in enumerate(zip(images, bboxes)):
            h, w = image.shape[0], image.shape[1]
            # place img in img4
            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            images_new[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            for x1, y1, bw, bh, label in bbox:
                x1_new = np.clip(x1 + padw, x1a, x2a)
                y1_new = np.clip(y1 + padh, y1a, y2a)
                x2_new = np.clip(x1 + padw + bw, x1a, x2a)
                y2_new = np.clip(y1 + padh + bh, y1a, y2a)
                bw_new = x2_new - x1_new
                bh_new = y2_new - y1_new
                if bw_new * bh_new > 1:
                    bboxes_new.append((x1_new, y1_new, bw_new, bh_new, label))
        return images_new, bboxes_new


class HSVAugment(NumpyOp):
    def __init__(self, inputs, outputs, mode="train", hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

    def forward(self, data, state):
        img = data
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img


class CategoryID2ClassID(NumpyOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(category, list(range(80)))}

    def forward(self, data, state):
        if data.size > 0:
            classes = np.array([self.mapping[int(x)] for x in data[:, -1]], dtype="float32")
            data[:, -1] = classes
        else:
            data = np.zeros(shape=(1, 5), dtype="float32")
        return data


class GTBox(NumpyOp):
    def __init__(self, inputs, outputs, image_size, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.image_size = image_size
        self.anchor_s = [(10, 13), (16, 30), (33, 23)]
        self.anchor_m = [(30, 61), (62, 45), (59, 119)]
        self.anchor_l = [(116, 90), (156, 198), (373, 326)]

    def forward(self, data, state):
        bbox = data[np.sum(data, 1) > 0]
        if bbox.size > 0:
            gt_sbbox = self._generate_target(data, anchors=self.anchor_s, feature_size=80)
            gt_mbbox = self._generate_target(data, anchors=self.anchor_m, feature_size=40)
            gt_lbbox = self._generate_target(data, anchors=self.anchor_l, feature_size=20)
        else:
            gt_sbbox = np.zeros((80, 80, 3, 6), dtype="float32")
            gt_mbbox = np.zeros((40, 40, 3, 6), dtype="float32")
            gt_lbbox = np.zeros((20, 20, 3, 6), dtype="float32")
        return gt_sbbox, gt_mbbox, gt_lbbox

    def _generate_target(self, bbox, anchors, feature_size, wh_threshold=4.0):
        object_boxes, label = bbox[:, :-1], bbox[:, -1]
        gt_bbox = np.zeros((feature_size, feature_size, 3, 6), dtype="float32")
        for object_idx, object_box in enumerate(object_boxes):
            for anchor_idx, anchor in enumerate(anchors):
                ratio = object_box[2:] / np.array(anchor, dtype="float32")
                match = np.max(np.maximum(ratio, 1 / ratio)) < wh_threshold
                if match:
                    center_feature_map = (object_box[:2] + object_box[2:] / 2) / self.image_size * feature_size
                    candidate_coords = self._get_candidate_coords(center_feature_map, feature_size)
                    for xc, yc in candidate_coords:
                        gt_bbox[yc, xc, anchor_idx][:4] = object_box  # use absoulte x1,y1,w,h
                        gt_bbox[yc, xc, anchor_idx][4] = 1.0
                        gt_bbox[yc, xc, anchor_idx][5] = label[object_idx]
        return gt_bbox

    @staticmethod
    def _get_candidate_coords(center_feature_map, feature_size):
        xc, yc = center_feature_map
        candidate_coords = [(int(xc), int(yc))]
        if xc % 1 < 0.5 and xc > 1:
            candidate_coords.append((int(xc) - 1, int(yc)))
        if xc % 1 >= 0.5 and xc < feature_size - 1:
            candidate_coords.append((int(xc) + 1, int(yc)))
        if yc % 1 < 0.5 and yc > 1:
            candidate_coords.append((int(xc), int(yc) - 1))
        if yc % 1 >= 0.5 and yc < feature_size - 1:
            candidate_coords.append((int(xc), int(yc) + 1))
        return candidate_coords


# Reuseable convolution
class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Standard bottleneck
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):  # ch_in, ch_out, shortcut
        super().__init__()
        self.cv1 = ConvBlock(c1, c2, 1)
        self.cv2 = ConvBlock(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        if self.add:
            out = out + x
        return out


# CSP Bottleneck with 3 convolutions
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):  # ch_in, ch_out, num_repeat, shortcut
        super().__init__()
        self.cv1 = ConvBlock(c1, c2 // 2)
        self.cv2 = ConvBlock(c1, c2 // 2)
        self.m = nn.Sequential(*[Bottleneck(c2 // 2, c2 // 2, shortcut) for _ in range(n)])
        self.cv3 = ConvBlock(c2, c2)

    def forward(self, x):
        out1 = self.cv1(x)
        out1 = self.m(out1)
        out2 = self.cv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.cv3(out)
        return out


# Focus wh information into c-space
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = ConvBlock(c1 * 4, c2, k)

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


# Spatial pyramid pooling layer used in YOLOv3-SPP
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        self.cv1 = ConvBlock(c1, c1 // 2, 1, 1)
        self.cv2 = ConvBlock(c1 // 2 * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], 1)
        x = self.cv2(x)
        return x


class YoloV5(nn.Module):
    def __init__(self, w, h, c, num_class=80):
        super().__init__()
        assert w % 32 == 0 and h % 32 == 0, "image width and height must be a multiple of 32"
        self.num_class = num_class
        self.focus = Focus(c, 32, 3)
        self.conv1 = ConvBlock(32, 64, 3, 2)
        self.c3_1 = C3(64, 64)
        self.conv2 = ConvBlock(64, 128, 3, 2)
        self.c3_2 = C3(128, 128, 3)
        self.conv3 = ConvBlock(128, 256, 3, 2)
        self.c3_3 = C3(256, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3, 2)
        self.spp = SPP(512, 512)
        self.c3_4 = C3(512, 512, shortcut=False)
        self.conv5 = ConvBlock(512, 256)
        self.up1 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.c3_5 = C3(512, 256, shortcut=False)
        self.up2 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.conv6 = ConvBlock(256, 128)
        self.c3_6 = C3(256, 128, shortcut=False)
        self.conv7 = ConvBlock(128, 128, 3, 2)
        self.c3_7 = C3(256, 256, shortcut=False)
        self.conv8 = ConvBlock(256, 256, 3, 2)
        self.c3_8 = C3(512, 512, shortcut=False)
        self.conv17 = nn.Conv2d(128, (num_class + 5) * 3, 1)
        self.conv20 = nn.Conv2d(256, (num_class + 5) * 3, 1)
        self.conv23 = nn.Conv2d(512, (num_class + 5) * 3, 1)
        self.stride = torch.tensor([8, 16, 32])
        self._initialize_detect_bias()

    def _initialize_detect_bias(self):
        for layer, stride in zip([self.conv17, self.conv20, self.conv23], self.stride):
            b = layer.bias.view(3, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / stride)**2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_class - 0.99))  # cls
            layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.c3_1(x)
        x = self.conv2(x)
        x_4 = self.c3_2(x)
        x = self.conv3(x_4)
        x_6 = self.c3_3(x)
        x = self.conv4(x_6)
        x = self.spp(x)
        x = self.c3_4(x)
        x_10 = self.conv5(x)
        x = self.up1(x_10)
        x = torch.cat([x, x_6], dim=1)
        x = self.c3_5(x)
        x_14 = self.conv6(x)
        x = self.up2(x_14)
        x = torch.cat([x, x_4], dim=1)
        x_17 = self.c3_6(x)
        x = self.conv7(x_17)
        x = torch.cat([x, x_14], dim=1)
        x_20 = self.c3_7(x)
        x = self.conv8(x_20)
        x = torch.cat([x, x_10], dim=1)
        x_23 = self.c3_8(x)
        out_17 = self.conv17(x_17)  # B, 255, h/8, w/8 - P3 stage
        out_20 = self.conv20(x_20)  # B, 255, h/16, w/16 - P4 stage
        out_23 = self.conv23(x_23)  # B, 255, h/32, w/32   - P5 stage
        out = [out_17, out_20, out_23]
        for i, x in enumerate(out):
            bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,20,20,3,85)
            out[i] = x.view(bs, 3, self.num_class + 5, ny, nx).permute(0, 3, 4, 1, 2).contiguous()
        return out


class RescaleTranspose(TensorOp):
    def forward(self, data, state):
        data = data.permute(0, 3, 1, 2) / 255
        return data


class DecodePred(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.strides = [8, 16, 32]
        self.num_anchor = 3
        self.width, self.height = 640, 640
        anchor_s = [(10, 13), (16, 30), (33, 23)]
        anchor_m = [(30, 61), (62, 45), (59, 119)]
        anchor_l = [(116, 90), (156, 198), (373, 326)]
        self.anchors = self.create_anchor(anchor_s, anchor_m, anchor_l, self.strides)
        self.grids = self.create_grid(self.strides, self.num_anchor)

    def build(self, framework, device):
        self.anchors = [anchor.to(device) for anchor in self.anchors]
        self.grids = [grid.to(device) for grid in self.grids]

    def create_grid(self, strides, num_anchor):
        grids = []
        for stride in strides:
            x_coor = [stride * i for i in range(self.width // stride)]
            y_coor = [stride * i for i in range(self.height // stride)]
            xx, yy = np.meshgrid(x_coor, y_coor)
            xx, yy = np.float32(xx), np.float32(yy)
            xx, yy = np.stack([xx] * num_anchor, axis=-1), np.stack([yy] * num_anchor, axis=-1)
            grids.append(torch.Tensor(np.stack([xx, yy], axis=-1)))
        return grids

    def create_anchor(self, anchor_s, anchor_m, anchor_l, strides):
        anchors = []
        for anchor, stride in zip([anchor_s, anchor_m, anchor_l], strides):
            feature_size_x, feature_size_y = self.width // stride, self.height // stride
            anchor = np.array(anchor, dtype="float32").reshape((1, 1, 3, 2))
            anchor = np.tile(anchor, [feature_size_y, feature_size_x, 1, 1])
            anchors.append(torch.Tensor(anchor))
        return anchors

    def forward(self, data, state):
        conv_sbbox = self.decode(data[0], self.grids[0], self.anchors[0], self.strides[0])
        conv_mbbox = self.decode(data[1], self.grids[1], self.anchors[1], self.strides[1])
        conv_lbbox = self.decode(data[2], self.grids[2], self.anchors[2], self.strides[2])
        return conv_sbbox, conv_mbbox, conv_lbbox

    def decode(self, conv_bbox, grid, anchor, stride):
        batch_size = conv_bbox.size(0)
        grid, anchor = torch.unsqueeze(grid, 0), torch.unsqueeze(anchor, 0)
        grid, anchor = grid.repeat(batch_size, 1, 1, 1, 1), anchor.repeat(batch_size, 1, 1, 1, 1)
        bbox_pred, conf_pred, cls_pred = torch.sigmoid(conv_bbox[..., 0:4]), conv_bbox[..., 4:5], conv_bbox[..., 5:]
        xcyc_pred, wh_pred = bbox_pred[..., 0:2], bbox_pred[..., 2:4]
        xcyc_pred = (xcyc_pred * 2 - 0.5) * stride + grid
        wh_pred = (wh_pred * 2)**2 * anchor
        x1y1_pred = xcyc_pred - wh_pred / 2
        result = torch.cat([x1y1_pred, wh_pred, conf_pred, cls_pred], dim=-1)
        return result


class ComputeLoss(LossOp):
    def __init__(self, inputs, outputs, img_size=640, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.BCEcls = nn.BCEWithLogitsLoss(reduction="none")
        self.BCEobj = nn.BCEWithLogitsLoss(reduction="none")
        self.img_size = img_size

    def forward(self, data, state):
        pred, true = data
        true_box, true_obj, true_class = torch.split(true, (4, 1, true.size(-1) - 5), dim=-1)
        pred_box, pred_obj, pred_class = torch.split(pred, (4, 1, pred.size(-1) - 5), dim=-1)
        num_classes = pred_class.size(-1)
        true_class = torch.squeeze(torch.nn.functional.one_hot(true_class.long(), num_classes), -2).half()
        box_scale = 2 - 1.0 * true_box[..., 2:3] * true_box[..., 3:4] / (self.img_size**2)
        iou = torch.unsqueeze(self.bbox_iou(pred_box, true_box, giou=True), -1)
        iou_loss = (1 - iou) * true_obj * box_scale
        conf_loss = self.BCEobj(pred_obj, true_obj)
        class_loss = true_obj * self.BCEcls(pred_class, true_class)
        iou_loss = torch.mean(torch.sum(iou_loss, (1, 2, 3, 4)))
        conf_loss = torch.mean(torch.sum(conf_loss, (1, 2, 3, 4)))
        class_loss = torch.mean(torch.sum(class_loss, (1, 2, 3, 4)))
        return iou_loss, conf_loss, class_loss

    @staticmethod
    def bbox_iou(bbox1, bbox2, giou=False, diou=False, ciou=False, epsilon=1e-7):
        b1x1, b1x2, b1y1, b1y2 = bbox1[..., 0], bbox1[..., 0] + bbox1[..., 2], bbox1[..., 1], bbox1[..., 1] + bbox1[..., 3]
        b2x1, b2x2, b2y1, b2y2  = bbox2[..., 0], bbox2[..., 0] + bbox2[..., 2], bbox2[..., 1], bbox2[..., 1] + bbox2[..., 3]
        # intersection area
        inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
                (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
        # union area
        w1, h1 = b1x2 - b1x1 + epsilon, b1y2 - b1y1 + epsilon
        w2, h2 = b2x2 - b2x1 + epsilon, b2y2 - b2y1 + epsilon
        union = w1 * h1 + w2 * h2 - inter + epsilon
        # iou
        iou = inter / union
        if giou or diou or ciou:
            cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)  # convex height
            if ciou or diou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + epsilon  # convex diagonal squared
                rho2 = ((b2x1 + b2x2 - b1x1 - b1x2)**2 + (b2y1 + b2y2 - b1y1 - b1y2)**2) / 4  # center distance squared
                if diou:
                    return iou - rho2 / c2  # DIoU
                elif ciou:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + epsilon))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + epsilon  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU


class PredictBox(TensorOp):
    def __init__(self, inputs, outputs, mode, width, height, max_outputs=500, conf_threshold=0.4):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.width = width
        self.height = height
        self.max_outputs = max_outputs
        self.conf_threshold = conf_threshold

    def forward(self, data, state):
        conv_sbbox, conv_mbbox, conv_lbbox = data
        batch_size = conv_sbbox.shape[0]
        final_results = []
        for idx in range(batch_size):
            pred_s, pred_m, pred_l = conv_sbbox[idx], conv_mbbox[idx], conv_lbbox[idx]
            pred_s, pred_m, pred_l = pred_s.view(-1, 85), pred_m.view(-1, 85), pred_l.view(-1, 85)
            preds = torch.cat([pred_s, pred_m, pred_l], dim=0)
            preds[:, 4] = torch.sigmoid(preds[:, 4])  # convert logits to confidence score
            preds = preds[preds[:, 4] > self.conf_threshold]  # filter by confidence
            selected_boxes_all_classes = torch.zeros(0, 6).to(conv_sbbox.device)
            if preds.size(0) > 0:
                classes = torch.argmax(preds[:, 5:], dim=-1)
                unique_classes = torch.unique(classes)
                for clss in unique_classes:
                    preds_cls = preds[classes == clss]
                    x1, y1, w, h = preds_cls[:, 0], preds_cls[:, 1], preds_cls[:, 2], preds_cls[:, 3]
                    x2, y2 = x1 + w, y1 + h
                    conf_score, label = preds_cls[:, 4], classes[classes == clss]
                    selected_bboxes = torch.stack([x1, y1, x2, y2, conf_score, label.to(x1.dtype)], dim=-1)
                    nms_keep = torchvision.ops.nms(selected_bboxes[:, :4], selected_bboxes[:, 4], iou_threshold=0.35)
                    selected_bboxes = selected_bboxes[nms_keep]
                    selected_boxes_all_classes = torch.cat([selected_boxes_all_classes, selected_bboxes], dim=0)
            # clamp values:
            x1_abs = selected_boxes_all_classes[:, 0].clamp(0, self.width)
            y1_abs = selected_boxes_all_classes[:, 1].clamp(0, self.height)
            width_abs = torch.min((selected_boxes_all_classes[:, 2] - x1_abs).clamp(0), self.width - x1_abs)
            height_abs = torch.min((selected_boxes_all_classes[:, 3] - y1_abs).clamp(0), self.height - y1_abs)
            labels_score, labels = selected_boxes_all_classes[:, 4], selected_boxes_all_classes[:, 5]
            results_single = [x1_abs, y1_abs, width_abs, height_abs, labels, labels_score, torch.ones_like(x1_abs)]
            results_single = torch.stack(results_single, dim=-1)
            # pad 0 to other rows to improve performance
            results_single = torch.nn.functional.pad(results_single,
                                                     (0, 0, 0, self.max_outputs - results_single.size(0)))
            final_results.append(results_single)
        final_results = torch.stack(final_results)
        return final_results


def lr_schedule_warmup(step, train_steps_epoch, init_lr):
    warmup_steps = train_steps_epoch * 3
    if step < warmup_steps:
        lr = init_lr / warmup_steps * step
    else:
        lr = init_lr
    return lr


def get_estimator(data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  epochs=200,
                  batch_size_per_gpu=32,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None):
    num_device = get_num_devices()
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir)
    train_ds = PreMosaicDataset(mscoco_ds=train_ds)
    batch_size = num_device * batch_size_per_gpu
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs=("image1", "image2", "image3", "image4"),
                      outputs=("image1", "image2", "image3", "image4"),
                      mode="train"),
            ReadImage(inputs="image", outputs="image", mode="eval"),
            LongestMaxSize(max_size=640,
                           image_in="image1",
                           bbox_in="bbox1",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image2",
                           bbox_in="bbox2",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image3",
                           bbox_in="bbox3",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image4",
                           bbox_in="bbox4",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image",
                           bbox_in="bbox",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="eval"),
            PadIfNeeded(min_height=640,
                        min_width=640,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="eval",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            CombineMosaic(inputs=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4"),
                          outputs=("image", "bbox"),
                          mode="train"),
            CenterCrop(height=640,
                       width=640,
                       image_in="image",
                       bbox_in="bbox",
                       bbox_params=BboxParams("coco", min_area=1.0),
                       mode="train"),
            Sometimes(
                HorizontalFlip(image_in="image",
                               bbox_in="bbox",
                               bbox_params=BboxParams("coco", min_area=1.0),
                               mode="train")),
            HSVAugment(inputs="image", outputs="image", mode="train"),
            ToArray(inputs="bbox", outputs="bbox", dtype="float32"),
            CategoryID2ClassID(inputs="bbox", outputs="bbox"),
            GTBox(inputs="bbox", outputs=("gt_sbbox", "gt_mbbox", "gt_lbbox"), image_size=640),
            Delete(keys=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4", "bbox"),
                   mode="train"),
            Delete(keys="image_id", mode="eval")
        ],
        pad_value=0)
    init_lr = 1e-2 / 64 * batch_size
    model = fe.build(
        lambda: YoloV5(w=640, h=640, c=3),
        optimizer_fn=lambda x: torch.optim.SGD(x, lr=init_lr, momentum=0.937, weight_decay=0.0005, nesterov=True),
        mixed_precision=True)
    network = fe.Network(ops=[
        RescaleTranspose(inputs="image", outputs="image"),
        ModelOp(model=model, inputs="image", outputs=("pred_s", "pred_m", "pred_l")),
        DecodePred(inputs=("pred_s", "pred_m", "pred_l"), outputs=("pred_s", "pred_m", "pred_l")),
        ComputeLoss(inputs=("pred_s", "gt_sbbox"), outputs=("sbbox_loss", "sconf_loss", "scls_loss")),
        ComputeLoss(inputs=("pred_m", "gt_mbbox"), outputs=("mbbox_loss", "mconf_loss", "mcls_loss")),
        ComputeLoss(inputs=("pred_l", "gt_lbbox"), outputs=("lbbox_loss", "lconf_loss", "lcls_loss")),
        Average(inputs=("sbbox_loss", "mbbox_loss", "lbbox_loss"), outputs="bbox_loss"),
        Average(inputs=("sconf_loss", "mconf_loss", "lconf_loss"), outputs="conf_loss"),
        Average(inputs=("scls_loss", "mcls_loss", "lcls_loss"), outputs="cls_loss"),
        Average(inputs=("bbox_loss", "conf_loss", "cls_loss"), outputs="total_loss"),
        PredictBox(width=640, height=640, inputs=("pred_s", "pred_m", "pred_l"), outputs="box_pred", mode="eval"),
        UpdateOp(model=model, loss_name="total_loss")
    ])
    traces = [
        MeanAveragePrecision(num_classes=80, true_key='bbox', pred_key='box_pred', mode="eval"),
        BestModelSaver(model=model, save_dir=model_dir, metric='mAP', save_best_mode="max")
    ]
    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_schedule_warmup(
                step, train_steps_epoch=np.ceil(len(train_ds) / batch_size), init_lr=init_lr)),
        4:
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(
                epoch, cycle_length=epochs - 3, init_lr=init_lr, min_lr=init_lr / 100, start=4))
    }
    traces.append(EpochScheduler(lr_schedule))
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=["bbox_loss", "conf_loss", "cls_loss"],
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator
