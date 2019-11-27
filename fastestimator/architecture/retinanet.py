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
"""RetinaNet implementation."""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models, regularizers

from fastestimator.op import TensorOp


def _classification_sub_net(num_classes, num_anchor=9):
    """Creates an object classification sub-network for the RetinaNet.

    Args:
        num_classes (int): number of classes.
        num_anchor (int, optional): number of anchor boxes. Defaults to 9.

    Returns:
        'Model' object: classification sub-network.
    """
    model = models.Sequential()
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(num_classes * num_anchor,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='sigmoid',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.initializers.constant(np.log(1 / 99))))
    model.add(layers.Reshape((-1, num_classes)))  # the output dimension is [batch, #anchor, #classes]
    return model


def _regression_sub_net(num_anchor=9):
    """Creates a regression sub-network for the RetinaNet.

    Args:
        num_anchor (int, optional): number of anchor boxes. Defaults to 9.

    Returns:
        'Model' object: regression sub-network.
    """
    model = models.Sequential()
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(4 * num_anchor,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Reshape((-1, 4)))  # the output dimension is [batch, #anchor, 4]
    return model


def RetinaNet(input_shape, num_classes, num_anchor=9):
    """Creates the RetinaNet. RetinaNet is composed of an FPN, a classification sub-network and a localization
    regression sub-network.

    Args:
        input_shape (tuple): shape of input image.
        num_classes (int): number of classes.
        num_anchor (int, optional): number of anchor boxes. Defaults to 9.

    Returns:
        'Model' object: RetinaNet.
    """
    inputs = tf.keras.Input(shape=input_shape)
    # FPN
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[80].name == "conv3_block4_out"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "conv4_block6_out"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "conv5_block3_out"
    C5 = resnet50.layers[-1].output
    P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C5)
    P5_upsampling = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C4)
    P4 = layers.Add()([P5_upsampling, P4])
    P4_upsampling = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C3)
    P3 = layers.Add()([P4_upsampling, P3])
    P6 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       name="P6",
                       kernel_regularizer=regularizers.l2(0.0001))(C5)
    P7 = layers.Activation('relu')(P6)
    P7 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       name="P7",
                       kernel_regularizer=regularizers.l2(0.0001))(P7)
    P5 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P5",
                       kernel_regularizer=regularizers.l2(0.0001))(P5)
    P4 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P4",
                       kernel_regularizer=regularizers.l2(0.0001))(P4)
    P3 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P3",
                       kernel_regularizer=regularizers.l2(0.0001))(P3)
    # classification subnet
    cls_subnet = _classification_sub_net(num_classes=num_classes, num_anchor=num_anchor)
    P3_cls = cls_subnet(P3)
    P4_cls = cls_subnet(P4)
    P5_cls = cls_subnet(P5)
    P6_cls = cls_subnet(P6)
    P7_cls = cls_subnet(P7)
    cls_output = layers.Concatenate(axis=-2)([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])
    # localization subnet
    loc_subnet = _regression_sub_net(num_anchor=num_anchor)
    P3_loc = loc_subnet(P3)
    P4_loc = loc_subnet(P4)
    P5_loc = loc_subnet(P5)
    P6_loc = loc_subnet(P6)
    P7_loc = loc_subnet(P7)
    loc_output = layers.Concatenate(axis=-2)([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    return tf.keras.Model(inputs=inputs, outputs=[cls_output, loc_output])


def _get_fpn_anchor_box(input_shape):
    """Returns the anchor boxes of the Feature Pyramid Net.

    Args:
        input_shape (tuple): shape of input image.

    Returns:
        array: numpy array with all anchor boxes.
    """
    assert len(input_shape) == 3
    h, w, _ = input_shape
    assert h % 32 == 0 and w % 32 == 0
    shapes = [(int(h / 8), int(w / 8))]  # P3
    num_pixel = [np.prod(shapes)]
    anchor_lengths = [32, 64, 128, 256, 512]
    for _ in range(4):  # P4 through P7
        shapes.append((int(np.ceil(shapes[-1][0] / 2)), int(np.ceil(shapes[-1][1] / 2))))
        num_pixel.append(np.prod(shapes[-1]))
    total_num_pixels = np.sum(num_pixel)
    anchorbox = np.zeros((9 * total_num_pixels, 4))
    anchor_length_multipliers = [2**(0.0), 2**(1 / 3), 2**(2 / 3)]
    aspect_ratios = [1.0, 2.0, 0.5]  #x:y
    anchor_idx = 0
    for shape, anchor_length in zip(shapes, anchor_lengths):
        p_h, p_w = shape
        base_y = 2**np.ceil(np.log2(h / p_h))
        base_x = 2**np.ceil(np.log2(w / p_w))
        for i in range(p_h):
            center_y = (i + 1 / 2) * base_y
            for j in range(p_w):
                center_x = (j + 1 / 2) * base_x
                for anchor_length_multiplier in anchor_length_multipliers:
                    area = (anchor_length * anchor_length_multiplier)**2
                    for aspect_ratio in aspect_ratios:
                        x1 = center_x - np.sqrt(area * aspect_ratio) / 2
                        y1 = center_y - np.sqrt(area / aspect_ratio) / 2
                        x2 = center_x + np.sqrt(area * aspect_ratio) / 2
                        y2 = center_y + np.sqrt(area / aspect_ratio) / 2
                        anchorbox[anchor_idx, 0] = x1
                        anchorbox[anchor_idx, 1] = y1
                        anchorbox[anchor_idx, 2] = x2 - x1
                        anchorbox[anchor_idx, 3] = y2 - y1
                        anchor_idx += 1
        if p_h == 1 and p_w == 1:  # the next level of 1x1 feature map is still 1x1, therefore ignore
            break
    return np.float32(anchorbox), np.int32(num_pixel) * 9


def _get_target(anchorbox, label, x1, y1, width, height):
    """Generates classification and localization ground-truths.

    Args:
        anchorbox (array): anchor boxes
        label (array): labels for each anchor box.
        x1 (array): x-coordinate of top left point of the box.
        y1 (array): y-coordinate of top left point of the box.
        width (array): width of the box.
        height (array): height of the box.

    Returns:
        array: classification groundtruths for each anchor box.
        array: localization groundtruths for each anchor box.
    """
    object_boxes = np.array([x1, y1, width, height]).T  # num_obj x 4
    ious = _get_iou(object_boxes, anchorbox)  # num_obj x num_anchor
    #now for each object in image, assign the anchor box with highest iou to them
    anchorbox_best_iou_idx = np.argmax(ious, axis=1)
    num_obj = ious.shape[0]
    for row in range(num_obj):
        ious[row, anchorbox_best_iou_idx[row]] = 0.99
    #next, begin the anchor box assignment based on iou
    anchor_to_obj_idx = np.argmax(ious, axis=0)  # num_anchor x 1
    anchor_best_iou = np.max(ious, axis=0)  # num_anchor x 1
    cls_gt = np.int32([label[idx] for idx in anchor_to_obj_idx])  # num_anchor x 1
    cls_gt[np.where(anchor_best_iou <= 0.4)] = -1  #background class
    cls_gt[np.where(np.logical_and(anchor_best_iou > 0.4, anchor_best_iou <= 0.5))] = -2  # ignore these examples
    #finally, get the selected localization coordinates
    anchor_has_object = np.where(cls_gt >= 0)
    box_anchor_obj = anchorbox[anchor_has_object]
    gt_object_idx = anchor_to_obj_idx[anchor_has_object]
    box_gt_obj = object_boxes[gt_object_idx]
    x1_gt, y1_gt, w_gt, h_gt = _get_loc_offset(box_gt_obj, box_anchor_obj)
    return cls_gt, x1_gt, y1_gt, w_gt, h_gt


def _get_loc_offset(box_gt, box_anchor):
    """Computes the offset of a groundtruth box and an anchor box.

    Args:
        box_gt (array): groundtruth box.
        box_anchor (array): anchor box.

    Returns:
        float: offset between x1 coordinate of the two boxes.
        float: offset between y1 coordinate of the two boxes.
        float: offset between width of the two boxes.
        float: offset between height of the two boxes.
    """
    gt_x1, gt_y1, gt_width, gt_height = np.split(box_gt, 4, axis=1)
    ac_x1, ac_y1, ac_width, ac_height = np.split(box_anchor, 4, axis=1)
    dx1 = (gt_x1 - ac_x1) / ac_width
    dy1 = (gt_y1 - ac_y1) / ac_height
    dwidth = np.log(gt_width / ac_width)
    dheight = np.log(gt_height / ac_height)
    return dx1, dy1, dwidth, dheight


def _get_iou(boxes1, boxes2):
    """Computes the value of intersection over union (IoU) of two array of boxes.

    Args:
        box1 (array): first boxes in N x 4
        box2 (array): second box in M x 4

    Returns:
        float: IoU value in N x M
    """
    x11, y11, w1, h1 = np.split(boxes1, 4, axis=1)
    x21, y21, w2, h2 = np.split(boxes2, 4, axis=1)
    x12 = x11 + w1
    y12 = y11 + h1
    x22 = x21 + w2
    y22 = y21 + h2
    xmin = np.maximum(x11, np.transpose(x21))
    ymin = np.maximum(y11, np.transpose(y21))
    xmax = np.minimum(x12, np.transpose(x22))
    ymax = np.minimum(y12, np.transpose(y22))
    inter_area = np.maximum((xmax - xmin + 1), 0) * np.maximum((ymax - ymin + 1), 0)
    area1 = (w1 + 1) * (h1 + 1)
    area2 = (w2 + 1) * (h2 + 1)
    iou = inter_area / (area1 + area2.T - inter_area)
    return iou


class PredictBox(TensorOp):
    """Convert network output to bounding boxes.
    """
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 input_shape=(512, 512, 3),
                 select_top_k=1000,
                 nms_max_outputs=100,
                 score_threshold=0.05):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.input_shape = input_shape
        self.select_top_k = select_top_k
        self.nms_max_outputs = nms_max_outputs
        self.score_threshold = score_threshold

        all_anchors, num_anchors_per_level = _get_fpn_anchor_box(input_shape=input_shape)
        self.all_anchors = tf.convert_to_tensor(all_anchors)
        self.num_anchors_per_level = num_anchors_per_level

    def forward(self, data, state):
        pred = []
        gt = []

        # extract max score and its class label
        cls_pred, deltas, label_gt, x1_gt, y1_gt, w_gt, h_gt = data
        labels = tf.cast(tf.argmax(cls_pred, axis=2), dtype=tf.int32)
        scores = tf.reduce_max(cls_pred, axis=2)

        # iterate over image
        for i in range(state['local_batch_size']):
            # GT for image
            image_gt_padded = tf.stack([
                tf.cast(i * tf.ones_like(label_gt[i]), dtype=tf.float32),
                x1_gt[i],
                y1_gt[i],
                w_gt[i],
                h_gt[i],
                tf.cast(label_gt[i], dtype=tf.float32)
            ],
                                       axis=1)
            keep_gt = label_gt[i] > 0
            image_gt = image_gt_padded[keep_gt]
            gt.append(image_gt)

            # split batch into images
            labels_per_image = labels[i]
            scores_per_image = scores[i]
            deltas_per_image = deltas[i]

            selected_deltas_per_image = tf.constant([], shape=(0, 4))
            selected_labels_per_image = tf.constant([], dtype=tf.int32)
            selected_scores_per_image = tf.constant([])
            selected_anchor_indices_per_image = tf.constant([], dtype=tf.int32)

            end_index = 0
            # iterate over each pyramid level
            for j in range(self.num_anchors_per_level.shape[0]):
                start_index = end_index
                end_index += self.num_anchors_per_level[j]
                anchor_indices = tf.range(start_index, end_index, dtype=tf.int32)

                level_scores = scores_per_image[start_index:end_index]
                level_deltas = deltas_per_image[start_index:end_index]
                level_labels = labels_per_image[start_index:end_index]

                # select top k
                if self.num_anchors_per_level[j] >= self.select_top_k:
                    top_k = tf.math.top_k(level_scores, self.select_top_k)
                    top_k_indices = top_k.indices
                else:
                    top_k_indices = tf.subtract(anchor_indices, [start_index])

                # combine all pyramid levels
                selected_deltas_per_image = tf.concat(
                    [selected_deltas_per_image, tf.gather(level_deltas, top_k_indices)], axis=0)
                selected_scores_per_image = tf.concat(
                    [selected_scores_per_image, tf.gather(level_scores, top_k_indices)], axis=0)
                selected_labels_per_image = tf.concat(
                    [selected_labels_per_image, tf.gather(level_labels, top_k_indices)], axis=0)
                selected_anchor_indices_per_image = tf.concat(
                    [selected_anchor_indices_per_image, tf.gather(anchor_indices, top_k_indices)], axis=0)

            # delta -> (x1, y1, w, h)
            selected_anchors_per_image = tf.gather(self.all_anchors, selected_anchor_indices_per_image)
            x1 = (selected_deltas_per_image[:, 0] * selected_anchors_per_image[:, 2]) + selected_anchors_per_image[:, 0]
            y1 = (selected_deltas_per_image[:, 1] * selected_anchors_per_image[:, 3]) + selected_anchors_per_image[:, 1]
            w = tf.math.exp(selected_deltas_per_image[:, 2]) * selected_anchors_per_image[:, 2]
            h = tf.math.exp(selected_deltas_per_image[:, 3]) * selected_anchors_per_image[:, 3]
            x2 = x1 + w
            y2 = y1 + h

            # nms
            # filter out low score, and perform nms
            boxes_per_image = tf.stack([y1, x1, y2, x2], axis=1)
            nms_indices = tf.image.non_max_suppression(boxes_per_image,
                                                       selected_scores_per_image,
                                                       self.nms_max_outputs,
                                                       score_threshold=self.score_threshold)

            nms_boxes = tf.gather(boxes_per_image, nms_indices)
            final_scores = tf.gather(selected_scores_per_image, nms_indices)
            final_labels = tf.cast(tf.gather(selected_labels_per_image, nms_indices), dtype=tf.float32)

            # clip bounding boxes to image size
            x1 = tf.clip_by_value(nms_boxes[:, 1], clip_value_min=0, clip_value_max=self.input_shape[1])
            y1 = tf.clip_by_value(nms_boxes[:, 0], clip_value_min=0, clip_value_max=self.input_shape[0])
            w = tf.clip_by_value(nms_boxes[:, 3], clip_value_min=0, clip_value_max=self.input_shape[1]) - x1
            h = tf.clip_by_value(nms_boxes[:, 2], clip_value_min=0, clip_value_max=self.input_shape[0]) - y1

            image_results = tf.stack([i * tf.ones_like(final_labels), x1, y1, w, h, final_labels, final_scores], axis=1)
            pred.append(image_results)

        return tf.concat(pred, axis=0), tf.concat(gt, axis=0)
