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
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np

def classification_sub_net(num_classes, num_anchor=9):
    model = models.Sequential()
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(num_classes * num_anchor,  kernel_size=3, strides=1, padding='same', activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.initializers.constant(np.log(1/99))))
    model.add(layers.Reshape((-1, num_classes)))  #the output dimension is [batch, #anchor, #classes]
    return model

def regression_sub_net(num_anchor=9):
    model = models.Sequential()
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Conv2D(4 * num_anchor,  kernel_size=3, strides=1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Reshape((-1, 4)))  #the output dimension is [batch, #anchor, 4]
    return model

def RetinaNet(input_shape, num_classes, num_anchor=9):
    inputs = tf.keras.Input(shape= input_shape)
    #FPN
    resnet50 = tf.keras.applications.ResNet50(weights= "imagenet", include_top= False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[80].name == "activation_21"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "activation_39"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "activation_48"
    C5 = resnet50.layers[-1].output
    P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampling = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C4)
    P4 = layers.Add()([P5_upsampling, P4])
    P4_upsampling = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C3)
    P3 = layers.Add()([P4_upsampling, P3])
    P6 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name="P6")(C5)
    P7 = layers.Activation('relu')(P6)
    P7 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name="P7")(P7)
    P5 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name="P5")(P5)
    P4 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name="P4")(P4)
    P3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name="P3")(P3)
    #classification subnet
    cls_subnet = classification_sub_net(num_classes=num_classes, num_anchor=num_anchor)
    P3_cls = cls_subnet(P3)
    P4_cls = cls_subnet(P4)
    P5_cls = cls_subnet(P5)
    P6_cls = cls_subnet(P6)
    P7_cls = cls_subnet(P7)
    cls_output = layers.Concatenate(axis=-2)([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])
    #localization subnet
    loc_subnet = regression_sub_net(num_anchor=num_anchor)
    P3_loc = loc_subnet(P3)
    P4_loc = loc_subnet(P4)
    P5_loc = loc_subnet(P5)
    P6_loc = loc_subnet(P6)
    P7_loc = loc_subnet(P7)
    loc_output = layers.Concatenate(axis=-2)([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    return tf.keras.Model(inputs=inputs, outputs=[cls_output, loc_output]) 

def get_fpn_anchor_box(input_shape):
    assert len(input_shape) == 3
    h, w, _ = input_shape
    assert h % 32 == 0 and w % 32 == 0
    shapes = [(int(h/8), int(w/8))] #P3
    num_pixel = np.prod(shapes)
    for _ in range(4): #P4 through P7
        shapes.append((int(np.ceil(shapes[-1][0]/2)), int(np.ceil(shapes[-1][1]/2))))
        num_pixel += np.prod(shapes[-1])
    anchorbox = np.zeros((9*num_pixel, 4))
    base_multipliers = [2 ** (0.0), 2 ** (1/3), 2 ** (2/3)]
    aspect_ratio_multiplier = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
    anchor_idx = 0
    for shape in shapes:
        p_h, p_w = shape
        base_y = 1/p_h
        base_x = 1/p_w
        for i in range(p_h):
            for j in range(p_w):
                for base_multiplier in base_multipliers:
                    for aspect_x, aspect_y in aspect_ratio_multiplier:
                        center_y = (i + 1/2) * base_y
                        center_x = (j + 1/2) * base_x
                        anchorbox[anchor_idx,0] = max(center_x - base_x * base_multiplier * aspect_x, 0.0)  #x1
                        anchorbox[anchor_idx,1] = max(center_y - base_y * base_multiplier * aspect_y, 0.0)  #y1
                        anchorbox[anchor_idx,2] = min(center_x + base_x * base_multiplier * aspect_x, 1.0)  #x2
                        anchorbox[anchor_idx,3] = min(center_y + base_y * base_multiplier * aspect_y, 1.0)  #y2
                        anchor_idx += 1
        if p_h == 1 and p_w == 1: #the next level of 1x1 feature map is still 1x1, therefore ignore
            break
    return np.float32(anchorbox)

def get_target(anchorbox, label, x1, y1, x2, y2, num_classes=10):
    num_anchor = anchorbox.shape[0]
    target_cls = np.zeros(shape=(num_anchor), dtype=np.int64)
    target_loc = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
    for _label, _x1, _y1, _x2, _y2 in zip(label, x1, y1, x2, y2):
        best_iou = 0.0
        for anchor_idx in range(num_anchor):
            iou = get_iou((_x1, _y1, _x2, _y2), anchorbox[anchor_idx])
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = anchor_idx
            if iou > 0.5:
                target_cls[anchor_idx] = _label
                target_loc[anchor_idx] = get_loc_offset((_x1, _y1, _x2, _y2), anchorbox[anchor_idx])
            elif iou >0.4:
                target_cls[anchor_idx] = -2 #ignore this example
            else:
                target_cls[anchor_idx] = -1 #background class
        if best_iou > 0 and best_iou < 0.5: #if gt has no >0.5 iou with any anchor
            target_cls[best_anchor_idx] = _label
            target_loc[best_anchor_idx] = get_loc_offset((_x1, _y1, _x2, _y2), anchorbox[best_anchor_idx])
    return target_cls, target_loc

def get_loc_offset(box_gt, box_anchor):
    gt_x1,gt_y1, gt_x2, gt_y2  = tuple(box_gt)
    ac_x1, ac_y1, ac_x2, ac_y2  = tuple(box_anchor)
    anchor_width = ac_x2 - ac_x1
    anchor_height = ac_y2 - ac_y1
    dx1 = (gt_x1 - ac_x1)/anchor_width
    dy1 = (gt_y1 - ac_y1)/anchor_height
    dx2 = (gt_x2 - ac_x2)/anchor_width
    dy2 = (gt_y2 - ac_y2)/anchor_height
    return dx1, dy1, dx2, dy2

def get_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2  = tuple(box1)
    b2_x1, b2_y1, b2_x2, b2_y2  = tuple(box2)
    xA = max(b1_x1, b2_x1)
    yA = max(b1_y1, b2_y1)
    xB = min(b1_x2, b2_x2)
    yB = min(b1_y2, b2_y2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        iou = 0
    else:
        box1Area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box2Area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        iou = interArea / (box1Area + box2Area - interArea)
    return iou