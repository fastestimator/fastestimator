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
"""U-Net example."""
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import UNet
from fastestimator.dataset.data import montgomery
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize, Rotate
from fastestimator.op.numpyop.univariate import Minmax, ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice


class CombineLeftRightMask(NumpyOp):
    """NumpyOp to combine left lung mask and right lung mask."""
    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        mask_left, mask_right = data
        data = mask_left + mask_right
        return data


def get_estimator(epochs=20,
                  batch_size=4,
                  max_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  log_steps=20,
                  data_dir=None):
    # step 1
    csv = montgomery.load_data(root_dir=data_dir)
    pipeline = fe.Pipeline(
        train_data=csv,
        eval_data=csv.split(0.2),
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", parent_path=csv.parent_path, outputs="image", read_flag='gray'),
            ReadImage(inputs="mask_left",
                      parent_path=csv.parent_path,
                      outputs="mask_left",
                      read_flag='gray',
                      mode='!infer'),
            ReadImage(inputs="mask_right",
                      parent_path=csv.parent_path,
                      outputs="mask_right",
                      read_flag='gray',
                      mode='!infer'),
            CombineLeftRightMask(inputs=("mask_left", "mask_right"), outputs="mask", mode='!infer'),
            Delete(keys=["mask_left", "mask_right"], mode='!infer'),
            Resize(image_in="image", width=512, height=512),
            Resize(image_in="mask", width=512, height=512, mode='!infer'),
            Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="mask", mode='train')),
            Sometimes(numpy_op=Rotate(
                image_in="image", mask_in="mask", limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, mode='train')),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask", mode='!infer')
        ])

    # step 2
    model = fe.build(model_fn=lambda: UNet(input_size=(512, 512, 1)),
                     optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0001),
                     model_names="lung_segmentation")
    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="pred_segment"),
        CrossEntropy(inputs=("pred_segment", "mask"), outputs="loss", form="binary"),
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
                             log_steps=log_steps,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
