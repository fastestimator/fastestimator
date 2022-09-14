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
import tensorflow as tf
from scipy.ndimage import center_of_mass
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.dataset.data.mscoco import load_data
from fastestimator.op.numpyop.multivariate import LongestMaxSize, PadIfNeeded, Resize
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp, RemoveIf
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.normalize import Normalize
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.trace import Trace


def resblock(inputs, filters):
    x = layers.Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters * 4, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    if inputs.shape[-1] != filters * 4:
        inputs = layers.Conv2D(filters=filters * 4, kernel_size=1, padding="same", use_bias=False)(inputs)
        inputs = layers.BatchNormalization(momentum=0.9)(inputs)
    x = x + inputs
    x = layers.ReLU()(x)
    return x


def transition_branch(x, c_out):
    num_branch_in, num_branch_out = len(x), len(c_out)
    x = x + [x[-1] for _ in range(num_branch_out - num_branch_in)]  # padding the list x with x[-1]
    x_new = []
    for idx, (x_i, c_i) in enumerate(zip(x, c_out)):
        if idx < num_branch_in:
            if x_i.shape[-1] != c_i:
                x_i = layers.Conv2D(filters=c_i, kernel_size=3, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.ReLU()(x_i)
        else:
            filter_in = x_i.shape[-1]
            for j in range(idx + 1 - num_branch_in):
                filter_out = c_i if j == idx - num_branch_in else filter_in
                x_i = layers.Conv2D(filters=filter_out, kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.ReLU()(x_i)
        x_new.append(x_i)
    return x_new


def basic_block(inputs, filters):
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)(inputs)
        inputs = layers.BatchNormalization(momentum=0.9)(inputs)
    x = x + inputs
    x = layers.ReLU()(x)
    return x


def branch_convs(x, num_block, c_out):
    x_new = []
    for x_i, num_conv, c in zip(x, num_block, c_out):
        for _ in range(num_conv):
            x_i = basic_block(x_i, c)
        x_new.append(x_i)
    return x_new


def fuse_convs(x, c_out):
    x_new = []
    for idx_out, planes_out in enumerate(c_out):
        x_new_i = []
        for idx_in, x_i in enumerate(x):
            if idx_in > idx_out:
                x_i = layers.Conv2D(filters=planes_out, kernel_size=1, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.UpSampling2D(size=(2**(idx_in - idx_out), 2**(idx_in - idx_out)))(x_i)
            elif idx_in < idx_out:
                for _ in range(idx_out - idx_in - 1):
                    x_i = layers.Conv2D(x_i.shape[-1], kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                    x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                    x_i = layers.ReLU()(x_i)
                x_i = layers.Conv2D(planes_out, kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
            x_new_i.append(x_i)
        x_new.append(layers.ReLU()(tf.math.add_n(x_new_i)))
    return x_new


def hrstage(x, num_module, num_block, c_out):
    x = transition_branch(x, c_out)
    for _ in range(num_module):
        x = branch_convs(x, num_block, c_out)
        x = fuse_convs(x, c_out)
    return x


def hrnet(input_shape=(256, 256, 3), num_classes=17):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = hrstage([x], num_module=1, num_block=(4, 4), c_out=(32, 64))
    x = hrstage(x, num_module=4, num_block=(4, 4, 4), c_out=(32, 64, 128))
    x = hrstage(x, num_module=3, num_block=(4, 4, 4, 4), c_out=(32, 64, 128, 256))
    x = layers.Conv2D(filters=num_classes, kernel_size=1, activation="sigmoid")(x[0])
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def collect_single_keypoint_ds(ds, cache_limit=None):
    images, keypoints, keypoint_bboxes = [], [], []
    for idx in range(len(ds)):
        item = ds[idx]
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
        num_channel = gt.shape[-1]
        for idx in range(num_channel):
            pred_s, gt_s = pred[..., idx], gt[..., idx]
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
                    # counted as mistake if the prediction center is off by more than 3 pixels
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
            Delete(keys=("keypoint", "keypoint_bbox", "kpscore", "image_down"))
        ])
    model = fe.build(model_fn=hrnet, optimizer_fn="adam")
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
