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
import numpy as np
import tensorflow as tf

from fastestimator.architecture.retinanet import RetinaNet, get_fpn_anchor_box, get_target
from fastestimator.dataset import svhn_data
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.preprocess import Minmax
from fastestimator.record.preprocess import ImageReader, Resize
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import NumpyOp, TensorOp


class String2List(NumpyOp):
    # this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data):
        for idx, elem in enumerate(data):
            data[idx] = np.array([int(x) for x in elem[1:-1].split(',')])
        return data


class RelativeCoordinate(NumpyOp):
    def forward(self, data):
        image, x1, y1, x2, y2 = data
        height, width = image.shape[0], image.shape[1]
        x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height
        return x1, y1, x2, y2


class GenerateTarget(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.anchorbox = get_fpn_anchor_box(input_shape=(64, 128, 3))

    def forward(self, data):
        label, x1, y1, x2, y2 = data
        target_cls, target_loc = get_target(self.anchorbox, label, x1, y1, x2, y2, num_classes=10)
        return target_cls, target_loc


class RetinaLoss(Loss):
    def focal_loss(self, cls_gt, cls_pred, num_classes, alpha=0.25, gamma=2.0):
        # cls_gt has shape [B, A], cls_pred is in [B, A, K]
        obj_idx = tf.where(tf.greater_equal(cls_gt, 0))  # index of object
        obj_bg_idx = tf.where(tf.greater_equal(cls_gt, -1))  # index of object and background
        cls_gt = tf.one_hot(cls_gt, num_classes)
        cls_gt = tf.gather_nd(cls_gt, obj_bg_idx)
        cls_pred = tf.gather_nd(cls_pred, obj_bg_idx)
        # getting the object count for each image in batch
        _, idx, count = tf.unique_with_counts(obj_bg_idx[:, 0])
        object_count = tf.gather_nd(count, tf.reshape(idx, (-1, 1)))
        object_count = tf.tile(tf.reshape(object_count, (-1, 1)), [1, num_classes])
        object_count = tf.cast(object_count, tf.float32)
        # reshape to the correct shape
        cls_gt = tf.reshape(cls_gt, (-1, 1))
        cls_pred = tf.reshape(cls_pred, (-1, 1))
        object_count = tf.reshape(object_count, (-1, 1))
        # compute the focal weight on each selected anchor box
        alpha_factor = tf.ones_like(cls_gt) * alpha
        alpha_factor = tf.where(tf.equal(cls_gt, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(cls_gt, 1), 1 - cls_pred, cls_pred)
        focal_weight = alpha_factor * focal_weight**gamma / object_count
        focal_loss = tf.losses.BinaryCrossentropy()(cls_gt, cls_pred, sample_weight=focal_weight)
        return focal_loss, obj_idx

    def smooth_l1(self, loc_gt, loc_pred, obj_idx):
        # loc_gt anf loc_pred has shape [B, A, 4]
        loc_gt = tf.gather_nd(loc_gt, obj_idx)
        loc_pred = tf.gather_nd(loc_pred, obj_idx)
        loc_gt = tf.reshape(loc_gt, (-1, 1))
        loc_pred = tf.reshape(loc_pred, (-1, 1))
        loc_diff = tf.abs(loc_gt - loc_pred)
        smooth_l1_loss = tf.where(tf.less(loc_diff, 1), 0.5 * loc_diff**2, loc_diff - 0.5)
        smooth_l1_loss = tf.reduce_mean(smooth_l1_loss)
        return smooth_l1_loss

    def calculate_loss(self, batch, state):
        cls_gt, loc_gt = batch["target_cls"], batch["target_loc"]
        cls_pred, loc_pred = batch["pred_cls"], batch["pred_loc"]
        focal_loss, obj_idx = self.focal_loss(cls_gt, cls_pred, num_classes=10)
        smooth_l1_loss = self.smooth_l1(loc_gt, loc_pred, obj_idx)
        return 40000 * focal_loss + smooth_l1_loss


class PredictBox(TensorOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.anchorbox = tf.convert_to_tensor(get_fpn_anchor_box(input_shape=(64, 128, 3)))
        self.anchor_w_h = tf.tile(self.anchorbox[:, 2:], [1, 2]) - tf.tile(self.anchorbox[:, :2], [1, 2])

    def forward(self, data):
        cls_pred, loc_pred = tuple(data)
        top_n = 10
        score_threshold = 0.2
        # convert the residual prediction to absolute prediction in (x1, y1, x2, y2)
        loc_pred = tf.map_fn(lambda x: x * self.anchor_w_h + self.anchorbox, elems=loc_pred, dtype=tf.float32,
                             back_prop=False)
        num_batch, num_anchor, _ = loc_pred.shape
        cls_best_score = tf.reduce_max(cls_pred, axis=-1)
        cls_best_class = tf.argmax(cls_pred, axis=-1)
        # select top n anchor boxes to proceed
        sorted_score = tf.sort(cls_best_score, direction='DESCENDING')
        top_n = tf.minimum(top_n, num_anchor)
        cls_best_score = tf.cond(
            tf.greater(num_anchor, top_n), lambda: tf.where(
                tf.greater_equal(cls_best_score, tf.tile(sorted_score[:, top_n - 1:top_n], [1, num_anchor])),
                cls_best_score, 0.0), lambda: cls_best_score)
        # Padded Nonmax suppression with threshold
        selected_indices_padded = tf.map_fn(
            lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True,
                                                          score_threshold=score_threshold).selected_indices,
            (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
        valid_outputs = tf.map_fn(
            lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True,
                                                          score_threshold=score_threshold).valid_outputs,
            (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
        # select output anchors after the NMS
        batch_index = tf.tile(tf.reshape(tf.range(num_batch), [-1, 1]), [1, top_n])
        selected_indices_padded = tf.stack([batch_index, selected_indices_padded], axis=-1)
        select_mask = tf.sequence_mask(valid_outputs, top_n)
        selected_anchors = tf.boolean_mask(selected_indices_padded, select_mask)
        # get the class and coordinates or output anchor
        loc_selected = tf.gather_nd(loc_pred, selected_anchors)
        cls_selected = tf.gather_nd(cls_best_class, selected_anchors)
        return cls_selected, loc_selected, valid_outputs


def get_estimator():
    # prepare data in disk
    train_csv, val_csv, path = svhn_data.load_data()
    writer = RecordWriter(
        train_data=train_csv, validation_data=val_csv, ops=[
            ImageReader(inputs="image", parent_path=path, outputs="image"),
            String2List(inputs=["label", "x1", "y1", "x2", "y2"], outputs=["label", "x1", "y1", "x2", "y2"]),
            RelativeCoordinate(inputs=("image", "x1", "y1", "x2", "y2"), outputs=("x1", "y1", "x2", "y2")),
            Resize(inputs="image", target_size=(64, 128), outputs="image"),
            GenerateTarget(inputs=("label", "x1", "y1", "x2", "y2"), outputs=("target_cls", "target_loc"))
        ])
    # prepare pipeline
    pipeline = Pipeline(batch_size=256, data=writer, ops=Minmax(inputs="image", outputs="image"), padded_batch=True)
    # prepare model
    model = build(keras_model=RetinaNet(input_shape=(64, 128, 3), num_classes=10), loss=RetinaLoss(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.0001))
    network = Network(ops=[
        ModelOp(inputs="image", model=model, outputs=["pred_cls", "pred_loc"]),
        PredictBox(outputs=("cls_selected", "loc_selected", "valid_outputs"), mode="eval")
    ])
    # prepare estimator
    estimator = Estimator(network=network, pipeline=pipeline, epochs=15, log_steps=20)
    return estimator
