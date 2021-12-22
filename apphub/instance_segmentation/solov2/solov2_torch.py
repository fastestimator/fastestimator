import os
import tempfile

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage.measurements import center_of_mass

import fastestimator as fe
from fastestimator.backend import to_tensor
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, Resize
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import L2Regularizaton
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.tensorop import LambdaOp, TensorOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.trace import Trace
from fastestimator.util import Suppressor, get_num_devices


def pad_with_coord(data):
    bs, _, h, w = data.shape
    x = torch.linspace(-1, 1, w, dtype=data.dtype, device=data.device).view(1, 1, 1, -1).expand(bs, 1, h, w)
    y = torch.linspace(-1, 1, h, dtype=data.dtype, device=data.device).view(1, 1, -1, 1).expand(bs, 1, h, w)
    data = torch.cat([data, x, y], axis=1)  # concatenate along channel dimension
    return data


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_C5 = nn.Conv2d(2048, 256, 1)
        self.conv2d_C4 = nn.Conv2d(1024, 256, 1)
        self.conv2d_C3 = nn.Conv2d(512, 256, 1)
        self.conv2d_C2 = nn.Conv2d(256, 256, 1)
        self.conv2d_P5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, C2, C3, C4, C5):
        # lateral conv
        P5 = self.conv2d_C5(C5)
        P5_up = nn.functional.interpolate(P5, scale_factor=2)
        P4 = self.conv2d_C4(C4)
        P4 = P4 + P5_up
        P4_up = nn.functional.interpolate(P4, scale_factor=2)
        P3 = self.conv2d_C3(C3)
        P3 = P3 + P4_up
        P3_up = nn.functional.interpolate(P3, scale_factor=2)
        P2 = self.conv2d_C2(C2)
        P2 = P2 + P3_up
        # fpn conv
        P5 = self.conv2d_P5(P5)
        P4 = self.conv2d_P4(P4)
        P3 = self.conv2d_P3(P3)
        P2 = self.conv2d_P2(P2)
        return P2, P3, P4, P5


class ConvNorm(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False)
        self.groupnorm = nn.GroupNorm(num_groups=groups, num_channels=c_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        return x


class MaskHead(nn.Module):
    def __init__(self, mid_ch=128, out_ch=256):
        super().__init__()
        self.convnorm_p2 = ConvNorm(256, mid_ch)
        self.convnorm_p3 = ConvNorm(256, mid_ch)
        self.convnorm_p4a = ConvNorm(256, mid_ch)
        self.convnorm_p4b = ConvNorm(mid_ch, mid_ch)
        self.convnorm_p5a = ConvNorm(258, mid_ch)
        self.convnorm_p5b = ConvNorm(mid_ch, mid_ch)
        self.convnorm_p5c = ConvNorm(mid_ch, mid_ch)
        self.convnorm_out = ConvNorm(mid_ch, out_ch, kernel_size=1)

    def forward(self, P2, P3, P4, P5):
        # first level
        P2 = nn.functional.relu(self.convnorm_p2(P2))
        # second level
        P3 = nn.functional.relu(self.convnorm_p3(P3))
        P3 = nn.functional.interpolate(P3, scale_factor=2)
        # third level
        P4 = nn.functional.relu(self.convnorm_p4a(P4))
        P4 = nn.functional.interpolate(P4, scale_factor=2)
        P4 = nn.functional.relu(self.convnorm_p4b(P4))
        P4 = nn.functional.interpolate(P4, scale_factor=2)
        # top level, add coordinate
        P5 = nn.functional.relu(self.convnorm_p5a(pad_with_coord(P5)))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        P5 = nn.functional.relu(self.convnorm_p5b(P5))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        P5 = nn.functional.relu(self.convnorm_p5c(P5))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        seg_outputs = nn.functional.relu(self.convnorm_out(P2 + P3 + P4 + P5))
        return seg_outputs


class HeadModel(nn.Module):
    def __init__(self, num_classes=80, chin_in=258, ch_feature=512, ch_kernel_out=256):
        super().__init__()
        self.convnorm_k1 = ConvNorm(chin_in, ch_feature)
        self.convnorm_c1 = ConvNorm(chin_in - 2, ch_feature)
        self.convnorm_k2 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c2 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_k3 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c3 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_k4 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c4 = ConvNorm(ch_feature, ch_feature)
        self.conv_kernel = nn.Conv2d(ch_feature, ch_kernel_out, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_kernel.weight.data, std=0.01)
        nn.init.zeros_(self.conv_kernel.bias.data)
        self.conv_cls = nn.Conv2d(ch_feature, num_classes, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_cls.weight.data, std=0.01)
        nn.init.constant_(self.conv_cls.bias.data, val=np.log(1 / 99))

    def forward(self, x):
        feature_kernel = x
        feature_cls = x[:, :-2, :, :]
        feature_kernel = nn.functional.relu(self.convnorm_k1(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c1(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k2(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c2(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k3(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c3(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k4(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c4(feature_cls))
        feature_kernel = self.conv_kernel(feature_kernel)
        feature_cls = self.conv_cls(feature_cls)
        return feature_kernel, feature_cls


class SoloV2Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool_p2 = nn.MaxPool2d(kernel_size=2)
        self.head_model = HeadModel(num_classes=num_classes)

    def forward(self, P2, P3, P4, P5):
        # applying maxpool first for P2
        P2 = self.maxpool_p2(P2)
        features = [P2, P3, P4, P5, P5]
        grid_sizes = [40, 36, 24, 16, 12]
        feat_kernel_list, feat_cls_list = [], []
        for feature, grid_size in zip(features, grid_sizes):
            feature = pad_with_coord(feature)
            feature = nn.functional.interpolate(feature,
                                                size=(grid_size, grid_size),
                                                mode='bilinear',
                                                align_corners=False)
            feat_kernel, feat_cls = self.head_model(feature)
            feat_kernel_list.append(feat_kernel)
            feat_cls_list.append(torch.sigmoid(feat_cls))
        return feat_cls_list, feat_kernel_list


class SoloV2(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        res50_layers = list(torchvision.models.resnet50(pretrained=True).children())
        self.res50_in_C2 = nn.Sequential(*(res50_layers[:5]))
        self.res50_C2_C3 = nn.Sequential(*(res50_layers[5]))
        self.res50_C3_C4 = nn.Sequential(*(res50_layers[6]))
        self.res50_C4_C5 = nn.Sequential(*(res50_layers[7]))
        self.fpn = FPN()
        self.mask_head = MaskHead()
        self.head = SoloV2Head(num_classes=num_classes)

    def forward(self, x):
        C2 = self.res50_in_C2(x)
        C3 = self.res50_C2_C3(C2)
        C4 = self.res50_C3_C4(C3)
        C5 = self.res50_C4_C5(C4)
        P2, P3, P4, P5 = self.fpn(C2, C3, C4, C5)
        feat_seg = self.mask_head(P2, P3, P4, P5)
        feat_cls_list, feat_kernel_list = self.head(P2, P3, P4, P5)
        return feat_seg, feat_cls_list, feat_kernel_list


class MergeMask(NumpyOp):
    def forward(self, data, state):
        data = np.stack(data, axis=-1)
        return data


class GetImageSize(NumpyOp):
    def forward(self, data, state):
        height, width, _ = data.shape
        return np.array([height, width], dtype="int32")


class Gt2Target(NumpyOp):
    def __init__(self,
                 inputs,
                 outputs,
                 mode=None,
                 im_size=1024,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
                 coord_sigma=0.05,
                 sampling_ratio=4.0):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.im_size = im_size
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(category, list(range(80)))}
        self.mask_size = int(im_size / sampling_ratio)

    def forward(self, data, state):
        masks, bboxes = data
        bboxes = np.array(bboxes, dtype="float32")
        masks = np.transpose(masks, [2, 0, 1])  # (H, W, #objects) -> (#objects, H, W)
        masks, bboxes = self.remove_empty_gt(masks, bboxes)
        # 91 classes -> 80 classes that starts from 1
        classes = np.array([self.mapping[int(x[-1])] + 1 for x in bboxes], dtype=np.int32)
        widths, heights = bboxes[:, 2], bboxes[:, 3]
        gt_match = []  # number of objects x (grid_idx, height_idx, width_idx, exist)
        for width, height, mask in zip(widths, heights, masks):
            object_match = []
            object_scale = np.sqrt(width * height)
            center_h, center_w = center_of_mass(mask)
            for grid_idx, ((lower_scale, upper_scale), num_grid) in enumerate(zip(self.scale_ranges, self.num_grids)):
                grid_matched = (object_scale >= lower_scale) & (object_scale <= upper_scale)
                if grid_matched:
                    w_delta, h_delta = 0.5 * width * self.coord_sigma, 0.5 * height * self.coord_sigma
                    coord_h, coord_w = int(center_h / mask.shape[0] * num_grid), int(center_w / mask.shape[1] * num_grid)
                    # each object will have some additional area of effect
                    top_box_extend = max(0, int((center_h - h_delta) / mask.shape[0] * num_grid))
                    down_box_extend = min(num_grid - 1, int((center_h + h_delta) / mask.shape[0] * num_grid))
                    left_box_extend = max(0, int((center_w - w_delta) / mask.shape[1] * num_grid))
                    right_box_extend = min(num_grid - 1, int((center_w + w_delta) / mask.shape[0] * num_grid))
                    # make sure the additional area of effect is at most 1 grid more
                    top_box_extend = max(top_box_extend, coord_h - 1)
                    down_box_extend = min(down_box_extend, coord_h + 1)
                    left_box_extend = max(left_box_extend, coord_w - 1)
                    right_box_extend = min(right_box_extend, coord_w + 1)
                    object_match.extend([(grid_idx, y, x, 1) for y in range(top_box_extend, down_box_extend + 1)
                                         for x in range(left_box_extend, right_box_extend + 1)])
            gt_match.append(object_match)
        gt_match = self.pad_match(gt_match)  #num_object x num_matches x [grid_idx, heihght_idx, width_idx, exist]
        return gt_match, masks, classes

    def pad_match(self, gt_match):
        max_num_matches = max([len(match) for match in gt_match])
        for match in gt_match:
            match.extend([(0, 0, 0, 0) for _ in range(max_num_matches - len(match))])
        return np.array(gt_match, dtype="int32")

    def remove_empty_gt(self, masks, bboxes):
        num_objects = masks.shape[0]
        non_empty_mask = np.sum(masks.reshape(num_objects, -1), axis=1) > 0
        return masks[non_empty_mask], bboxes[non_empty_mask]


class NormalizePermute(TensorOp):
    def __init__(self, inputs, outputs, mean, std, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.mean = to_tensor(np.array(mean, dtype="float32"), "torch")
        self.std = to_tensor(np.array(std, dtype="float32"), "torch")

    def build(self, framework, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def forward(self, data, state):
        data = (data / 255 - self.mean) / self.std
        return data.permute((0, 3, 1, 2))  # channel first


class PointsNMS(TensorOp):
    def forward(self, data, state):
        feat_cls_list = [self.points_nms(x) for x in data]
        return feat_cls_list

    def points_nms(self, x):
        x_max_pool = nn.functional.max_pool2d(x, kernel_size=2, stride=1, padding=1)[..., :-1, :-1]
        x = torch.where(x == x_max_pool, x, torch.zeros_like(x))
        return x


class Solov2Loss(TensorOp):
    def __init__(self, level, grid_dim, inputs, outputs, mode=None, num_class=80):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level
        self.grid_dim = grid_dim
        self.num_class = num_class

    def forward(self, data, state):
        cls_losses, seg_losses = [], []
        for mask, cls, match, feat_seg, feat_cls, kernel in zip(*data):
            cls_loss, grid_object_map = self.get_cls_loss(cls, feat_cls, match)
            seg_loss = self.get_seg_loss(mask, feat_seg, kernel, grid_object_map)
            cls_losses.append(cls_loss)
            seg_losses.append(seg_loss)
        return torch.stack(cls_losses), torch.stack(seg_losses)

    def get_seg_loss(self, mask, feat_seg, kernel, grid_object_map):
        indices = torch.where(grid_object_map[..., 0] > 0)
        object_indices = grid_object_map[indices][:, 1].long()
        mask_gt = mask[object_indices].type(kernel.dtype)
        active_kernel = kernel.permute(1, 2, 0)[indices]
        seg_preds = torch.mm(active_kernel, feat_seg.view(feat_seg.size(0), -1)).view(mask_gt.shape)
        loss = self.dice_loss(seg_preds, mask_gt)
        return loss

    def dice_loss(self, pred, gt):
        pred = torch.sigmoid(pred)
        a = torch.sum(pred * gt)
        b = torch.sum(pred * pred) + 0.001
        c = torch.sum(gt * gt) + 0.001
        dice = 2 * a / (b + c)
        return 1 - torch.where(dice > 0.0, dice, dice - dice + 1)

    def get_cls_loss(self, cls_gt, feat_cls, match):
        cls_gt = cls_gt.type(feat_cls.dtype)
        match, cls_gt = match[cls_gt > 0], cls_gt[cls_gt > 0]  # remove the padded object
        feat_cls_gts_raw = torch.stack(
            [self.assign_cls_feat(match_single, cls_gt_single) for match_single, cls_gt_single in zip(match, cls_gt)])
        # reduce the gt for all objects into single grid
        # TODO: if there are multiple objects overlapping on same grid point, randomly choose one
        feat_cls_gts, object_idx = torch.max(feat_cls_gts_raw, dim=0)
        grid_object_map = torch.stack([feat_cls_gts, object_idx.type(feat_cls_gts.dtype)], dim=-1)
        # classification loss
        feat_cls_gts = nn.functional.one_hot(feat_cls_gts.long(), num_classes=self.num_class + 1)[..., 1:]
        cls_loss = self.focal_loss(feat_cls.permute(1, 2, 0).reshape(-1), feat_cls_gts.type(feat_cls.dtype).view(-1))
        return cls_loss, grid_object_map

    def focal_loss(self, pred, gt, alpha=0.25, gamma=2.0):
        anchor_obj_count = torch.count_nonzero(gt).type(pred.dtype)
        alpha_factor = torch.ones_like(gt) * alpha
        alpha_factor = torch.where(gt == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = torch.where(gt == 1, 1 - pred, pred)
        focal_weight = alpha_factor * focal_weight**gamma / (anchor_obj_count + 1)
        cls_loss = nn.functional.binary_cross_entropy(input=pred,
                                                      target=gt,
                                                      weight=focal_weight.detach(),
                                                      reduction="sum")
        return cls_loss

    def assign_cls_feat(self, grid_match_info, cls_gt_obj):
        match_bool = torch.logical_and(grid_match_info.sum(-1) > 0, grid_match_info[:, 0] == self.level)
        grid_match_info = grid_match_info[match_bool]
        grid_indices = grid_match_info[:, 1:3]
        feat_cls_gt = torch.zeros(self.grid_dim, self.grid_dim, dtype=cls_gt_obj.dtype, device=cls_gt_obj.device)
        for row, col in grid_indices:
            feat_cls_gt[row, col] = cls_gt_obj
        return feat_cls_gt


class CombineLoss(TensorOp):
    def forward(self, data, state):
        l_c1, l_s1, l_c2, l_s2, l_c3, l_s3, l_c4, l_s4, l_c5, l_s5 = data
        cls_losses = torch.sum(torch.stack([l_c1, l_c2, l_c3, l_c4, l_c5], dim=-1), dim=-1)
        seg_losses = torch.sum(torch.stack([l_s1, l_s2, l_s3, l_s4, l_s5], dim=-1), dim=-1)
        mean_cls_loss, mean_seg_loss = torch.mean(cls_losses), torch.mean(seg_losses) * 3
        return mean_cls_loss + mean_seg_loss, mean_cls_loss, mean_seg_loss


class Predict(TensorOp):
    def __init__(self, inputs, outputs, mode=None, score_threshold=0.1, strides=[8.0, 8.0, 16.0, 32.0, 32.0]):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.score_threshold = score_threshold
        self.strides = strides
        self.post_nms_k = 100

    def forward(self, data, state):
        feat_seg, feat_cls_list, feat_kernel_list = data
        strides = [
            torch.full((x.size(2) * x.size(3), ), s, device=x.device) for s, x in zip(self.strides, feat_cls_list)
        ]
        batch_size, num_class = feat_cls_list[0].size(0), feat_cls_list[0].size(1)
        kernel_dim = feat_kernel_list[0].size(1)
        feat_cls = torch.cat([x.view(batch_size, num_class, -1) for x in feat_cls_list], axis=-1)
        feat_kernel = torch.cat([x.view(batch_size, kernel_dim, -1) for x in feat_kernel_list], axis=-1)
        strides = torch.cat(strides, axis=0)
        seg_preds, cate_scores, cate_labels = [], [], []
        for feat_cls_s, feat_seg_s, feat_kernel_s in zip(feat_cls, feat_seg, feat_kernel):
            seg_pred, cate_score, cate_label = self.predict_sample(feat_cls_s, feat_seg_s, feat_kernel_s, strides)
            seg_preds.append(seg_pred)
            cate_scores.append(cate_score)
            cate_labels.append(cate_label)
        return torch.stack(seg_preds), torch.stack(cate_scores), torch.stack(cate_labels)

    def predict_sample(self, cate_preds, seg_preds, kernel_preds, strides):
        cate_preds = cate_preds.transpose(1, 0)
        kernel_preds = kernel_preds.transpose(1, 0)
        # first filter class prediction by score_threshold
        select_row, select_col = torch.where(cate_preds > self.score_threshold)
        cate_labels = select_col
        kernel_preds = kernel_preds[select_row]
        cate_scores, strides = cate_preds[select_row, select_col], strides[select_row]
        # next calculate the mask
        kernel_preds = kernel_preds[..., None, None]  # c_out, c_in, k_h, k_w
        seg_preds = seg_preds.unsqueeze(0)  # B, C, H, W
        if select_row.size(0) > 0:
            seg_preds = torch.sigmoid(nn.functional.conv2d(seg_preds, kernel_preds))[0]
        else:
            seg_preds = torch.zeros(0, seg_preds.size(2), seg_preds.size(3), device=seg_preds.device)
        seg_masks = torch.where(seg_preds > 0.5, 1.0, 0.0)
        # then filter masks based on strides
        mask_sum = seg_masks.sum([1, 2])
        seg_preds, seg_masks = seg_preds[mask_sum > strides], seg_masks[mask_sum > strides]
        cate_labels, cate_scores = cate_labels[mask_sum > strides], cate_scores[mask_sum > strides]
        mask_sum = mask_sum[mask_sum > strides]
        # scale the category score by mask confidence then matrix nms
        mask_scores = torch.sum(seg_preds * seg_masks, dim=[1, 2]) / mask_sum
        cate_scores = cate_scores * mask_scores
        if seg_preds.size(0) > 0:
            seg_preds, cate_scores, cate_labels = self.matrix_nms(seg_preds, seg_masks, cate_labels, cate_scores, mask_sum)
        # pad output for batch shape consistency
        num_selected = seg_preds.size(0)
        seg_preds = nn.functional.pad(seg_preds, pad=(0, 0, 0, 0, 0, self.post_nms_k - num_selected))
        cate_scores = nn.functional.pad(cate_scores, pad=(0, self.post_nms_k - num_selected))
        cate_labels = nn.functional.pad(cate_labels, pad=(0, self.post_nms_k - num_selected))
        return seg_preds, cate_scores, cate_labels

    def matrix_nms(self, seg_preds, seg_masks, cate_labels, cate_scores, mask_sum, pre_nms_k=500):
        # first select top k category scores
        num_selected = min(pre_nms_k, cate_scores.size(0))
        indices = torch.argsort(cate_scores, descending=True)[:num_selected]
        seg_preds, seg_masks, mask_sum = seg_preds[indices], seg_masks[indices], mask_sum[indices]
        cate_labels, cate_scores = cate_labels[indices], cate_scores[indices]
        # calculate iou between different masks
        seg_masks = seg_masks.view(seg_masks.size(0), -1)
        intersection = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        mask_sum = mask_sum.unsqueeze(0).expand(num_selected, -1)
        union = mask_sum + mask_sum.transpose(1, 0) - intersection
        iou = intersection / union
        iou = torch.triu(iou, diagonal=1)
        # iou decay and compensation
        labels_match = cate_labels.unsqueeze(0).expand(num_selected, -1)
        labels_match = torch.where(labels_match == labels_match.transpose(1, 0), 1.0, 0.0)
        labels_match = torch.triu(labels_match, diagonal=1)
        decay_iou = iou * labels_match  # iou with any object from same class
        compensate_iou, _ = decay_iou.max(dim=0)
        compensate_iou = compensate_iou.unsqueeze(1).expand(-1, num_selected)
        # matrix nms
        decay_coefficient, _ = torch.min(torch.exp(-2 * decay_iou**2) / torch.exp(-2 * compensate_iou**2), dim=0)
        cate_scores = cate_scores * decay_coefficient
        cate_scores = torch.where(cate_scores >= 0.05, cate_scores, cate_scores - cate_scores)
        num_selected = min(self.post_nms_k, cate_scores.size(0))
        indices = torch.argsort(cate_scores, descending=True)[:num_selected]
        seg_preds, cate_scores, cate_labels = seg_preds[indices], cate_scores[indices], cate_labels[indices]
        return seg_preds, cate_scores, cate_labels


def lr_schedule_warmup(step, init_lr):
    if step < 1000:
        lr = init_lr / 1000 * step
    else:
        lr = init_lr
    return lr


class COCOMaskmAP(Trace):
    def __init__(self, data_dir, inputs=None, outputs="mAP", mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        with Suppressor():
            self.coco_gt = COCO(os.path.join(data_dir, "MSCOCO2017", "annotations", "instances_val2017.json"))
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(list(range(80)), category)}

    def on_epoch_begin(self, data):
        self.results = []

    def on_batch_end(self, data):
        seg_preds, = data['seg_preds'].numpy(),
        cate_scores, cate_labels = data['cate_scores'].numpy(), data['cate_labels'].numpy()
        image_ids, imsizes = data['image_id'].numpy(), data['imsize'].numpy()
        for seg_pred, cate_score, cate_label, image_id, imsize in zip(seg_preds, cate_scores, cate_labels, image_ids, imsizes):
            # remove the padded data due to batching
            indices = cate_score > 0.01
            seg_pred, cate_score, cate_label = seg_pred[indices], cate_score[indices], cate_label[indices]
            if seg_pred.shape[0] == 0:
                continue
            seg_pred = np.transpose(seg_pred, axes=(1, 2, 0))  # [H, W, #objects]
            # remove the padded data due to image resize
            mask_h, mask_w, num_obj = seg_pred.shape
            image_h, image_w = 4 * mask_h, 4 * mask_w
            seg_pred = cv2.resize(seg_pred, (image_w, image_h))
            if num_obj == 1:
                seg_pred = seg_pred[..., np.newaxis]  # when there's only single object, resize will remove the channel
            ori_h, ori_w = imsize
            scale_ratio = min(image_h / ori_h, image_w / ori_w)
            pad_h, pad_w = image_h - scale_ratio * ori_h, image_w - scale_ratio * ori_w
            h_start, h_end = round(pad_h / 2), image_h - round(pad_h / 2)
            w_start, w_end = round(pad_w / 2), image_w - round(pad_w / 2)
            seg_pred = seg_pred[h_start:h_end, w_start:w_end, :]
            # now reshape to original shape
            seg_pred = cv2.resize(seg_pred, (ori_w, ori_h))
            if num_obj == 1:
                seg_pred = seg_pred[..., np.newaxis]  # when there's only single object, resize will remove the channel
            seg_pred = np.transpose(seg_pred, [2, 0, 1])  # [#objects, H, W]
            seg_pred = np.uint8(np.where(seg_pred > 0.5, 1, 0))
            for seg, score, label in zip(seg_pred, cate_score, cate_label):
                result = {
                    "image_id": image_id,
                    "category_id": self.mapping[label],
                    "score": score,
                    "segmentation": mask_util.encode(np.array(seg[..., np.newaxis], order='F'))[0]
                }
                self.results.append(result)
        return data

    def on_epoch_end(self, data):
        mAP = 0.0
        if self.results:
            with Suppressor():
                coco_results = self.coco_gt.loadRes(self.results)
                cocoEval = COCOeval(self.coco_gt, coco_results, 'segm')
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                mAP = cocoEval.stats[0]
        data.write_with_log(self.outputs[0], mAP)


def get_estimator(data_dir=None,
                  epochs=12,
                  batch_size_per_gpu=8,
                  im_size=1024,
                  model_dir=tempfile.mkdtemp(),
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None):
    assert im_size % 32 == 0, "im_size must be a multiple of 32"
    num_device = get_num_devices()
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir, load_masks=True)
    batch_size = num_device * batch_size_per_gpu
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        test_data=val_ds,
        batch_size=num_device * batch_size_per_gpu,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            MergeMask(inputs="mask", outputs="mask"),
            GetImageSize(inputs="image", outputs="imsize", mode="test"),
            LongestMaxSize(max_size=im_size, image_in="image", mask_in="mask", bbox_in="bbox", bbox_params="coco"),
            PadIfNeeded(min_height=im_size,
                        min_width=im_size,
                        image_in="image",
                        mask_in="mask",
                        bbox_in="bbox",
                        bbox_params="coco",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
            Sometimes(HorizontalFlip(image_in="image", mask_in="mask", bbox_in="bbox", bbox_params="coco",
                                     mode="train")),
            Resize(height=im_size // 4, width=im_size // 4, image_in='mask'),  # downscale mask for memory efficiency
            Gt2Target(inputs=("mask", "bbox"), outputs=("gt_match", "mask", "classes")),
            Delete(keys="bbox"),
            Delete(keys="image_id", mode="!test")
        ],
        pad_value=0,
        num_process=8 * num_device)
    init_lr = 1e-2 / 16 * batch_size
    model = fe.build(model_fn=SoloV2,
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=init_lr, momentum=0.9, weight_decay=1e-4))
    network = fe.Network(ops=[
        NormalizePermute(inputs="image", outputs="image", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ModelOp(model=model, inputs="image", outputs=("feat_seg", "feat_cls_list", "feat_kernel_list")),
        LambdaOp(fn=lambda x: x, inputs="feat_cls_list", outputs=("cls1", "cls2", "cls3", "cls4", "cls5")),
        LambdaOp(fn=lambda x: x, inputs="feat_kernel_list", outputs=("k1", "k2", "k3", "k4", "k5")),
        Solov2Loss(0, 40, inputs=("mask", "classes", "gt_match", "feat_seg", "cls1", "k1"), outputs=("l_c1", "l_s1")),
        Solov2Loss(1, 36, inputs=("mask", "classes", "gt_match", "feat_seg", "cls2", "k2"), outputs=("l_c2", "l_s2")),
        Solov2Loss(2, 24, inputs=("mask", "classes", "gt_match", "feat_seg", "cls3", "k3"), outputs=("l_c3", "l_s3")),
        Solov2Loss(3, 16, inputs=("mask", "classes", "gt_match", "feat_seg", "cls4", "k4"), outputs=("l_c4", "l_s4")),
        Solov2Loss(4, 12, inputs=("mask", "classes", "gt_match", "feat_seg", "cls5", "k5"), outputs=("l_c5", "l_s5")),
        CombineLoss(inputs=("l_c1", "l_s1", "l_c2", "l_s2", "l_c3", "l_s3", "l_c4", "l_s4", "l_c5", "l_s5"),
                    outputs=("total_loss", "cls_loss", "seg_loss")),
        L2Regularizaton(inputs="total_loss", outputs="total_loss_l2", model=model, beta=1e-4, mode="train"),
        UpdateOp(model=model, loss_name="total_loss_l2"),
        PointsNMS(inputs="feat_cls_list", outputs="feat_cls_list", mode="test"),
        Predict(inputs=("feat_seg", "feat_cls_list", "feat_kernel_list"),
                outputs=("seg_preds", "cate_scores", "cate_labels"),
                mode="test")
    ])
    train_steps_epoch = int(np.ceil(len(train_ds) / batch_size))
    lr_schedule = {
        1:
        LRScheduler(model=model, lr_fn=lambda step: lr_schedule_warmup(step, init_lr=init_lr)),
        2:
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(step,
                                            cycle_length=train_steps_epoch * (epochs - 1),
                                            init_lr=init_lr,
                                            min_lr=init_lr / 100,
                                            start=train_steps_epoch))
    }
    traces = [
        EpochScheduler(lr_schedule),
        COCOMaskmAP(data_dir=data_dir,
                    inputs=("seg_preds", "cate_scores", "cate_labels", "image_id", "imsize"),
                    mode="test"),
        BestModelSaver(model=model, save_dir=model_dir, metric="total_loss")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=("cls_loss", "seg_loss"),
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator
