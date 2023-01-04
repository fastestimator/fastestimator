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
"""Line Search example."""
import tempfile

import cv2
import numpy as np
import torch
from skimage import measure

import fastestimator as fe
from fastestimator.architecture.pytorch import UNet
from fastestimator.dataset.data import montgomery
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop import LambdaOp as NLambdaOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import Resize, Rotate
from fastestimator.op.numpyop.univariate import Binarize, ChannelTranspose, Minmax, ReadImage
from fastestimator.op.tensorop import LambdaOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice


class BoundingBoxFromMask(fe.op.numpyop.NumpyOp):
    """
    Args:
        inputs: Key(s) of  masks to be combined.
        outputs: Key(s) into which to write the combined masks.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer".
    """
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        y_true = data
        if len(y_true.shape) == 2:
            y_true = np.expand_dims(y_true, axis=-1)
        mask = np.zeros_like(y_true)

        blobs, n_blob = measure.label(y_true[:, :, 0], background=0, return_num=True)
        for b in range(1, n_blob + 1):
            blob_mask = blobs == b
            coords = np.argwhere(blob_mask)
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)
            box = [y1, x1, y2, x2, 0]
            mask[box[1]:box[3] + 1, box[0]:box[2] + 1, box[4]] = 1

        return mask


def get_estimator(epochs=60,
                  batch_size=8,
                  train_steps_per_epoch=500,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir="/raid/shared_data",
                  line_degree=45):
    csv_dataset = montgomery.load_data(root_dir=data_dir)
    pipeline = fe.Pipeline(
        train_data=csv_dataset,
        eval_data=csv_dataset.split(0.2),
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", parent_path=csv_dataset.parent_path, outputs="image", color_flag='gray'),
            ReadImage(inputs="mask_left", parent_path=csv_dataset.parent_path, outputs="mask_left", color_flag='gray'),
            ReadImage(inputs="mask_right", parent_path=csv_dataset.parent_path, outputs="mask_right",
                      color_flag='gray'),
            NLambdaOp(fn=lambda x, y: x + y, inputs=("mask_left", "mask_right"), outputs="mask"),
            Delete(keys=("mask_left", "mask_right")),
            Resize(image_in="image", width=512, height=512),
            Resize(image_in="mask", width=512, height=512),
            Sometimes(numpy_op=Rotate(image_in="image",
                                      mask_in="mask",
                                      limit=(-line_degree, line_degree),
                                      border_mode=cv2.BORDER_CONSTANT,
                                      mode='train')),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask"),
            Binarize(inputs="mask", outputs="mask", threshold=0.5),
            BoundingBoxFromMask(inputs="mask", outputs="box_mask"),
            ChannelTranspose(inputs=["image", "mask", "box_mask"], outputs=["image", "mask", "box_mask"])
        ])
    # step 2
    model = fe.build(model_fn=lambda: UNet(input_size=(1, 512, 512)),
                     optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=0.0001))
    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="pred_segment"),
        LambdaOp(fn=lambda x: torch.max(x, axis=2)[0], inputs="pred_segment", outputs="pred_x"),
        LambdaOp(fn=lambda x: torch.max(x, axis=2)[0], inputs="box_mask", outputs="mask_x"),
        LambdaOp(fn=lambda x: torch.max(x, axis=3)[0], inputs="pred_segment", outputs="pred_y"),
        LambdaOp(fn=lambda x: torch.max(x, axis=3)[0], inputs="box_mask", outputs="mask_y"),
        CrossEntropy(inputs=("pred_x", "mask_x"), outputs="loss_x", form="binary"),
        CrossEntropy(inputs=("pred_y", "mask_y"), outputs="loss_y", form="binary"),
        CrossEntropy(inputs=("pred_segment", "mask"), outputs="ce", form="binary"),  # To track model performance ONLY
        LambdaOp(fn=lambda x, y: x + y, inputs=("loss_x", "loss_y"), outputs="loss", mode="!infer"),
        UpdateOp(model=model, loss_name="loss")
    ])

    # step 3
    traces = [
        Dice(true_key="mask", pred_key="pred_segment"),
        BestModelSaver(model=model, save_dir=save_dir, metric='Dice', save_best_mode='max')
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch,
                             monitor_names='ce')
    return estimator
