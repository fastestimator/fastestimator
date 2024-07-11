"""MobileNetV3 with DeepLabV3 example."""
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

import fastestimator as fe
import torchvision.models as models

from fastestimator.dataset.data import montgomery
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize, Rotate
from fastestimator.op.numpyop.univariate import Minmax, ReadImage, Reshape, ChannelTranspose
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice


class CombineLeftRightMask(NumpyOp):
    """NumpyOp to combine left lung mask and right lung mask."""
    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        mask_left, mask_right = data
        data = np.maximum(mask_left, mask_right)
        return data


# class RepeatChannel(NumpyOp): # new local channel op for image formatting
#     """NumpyOp to repeat a single channel grayscale image to 3 channels."""
#     def forward(self, data: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
#         return np.repeat(data, 3, axis=0)


def get_estimator(epochs=20,
                  batch_size=4,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
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
            ReadImage(inputs="image", parent_path=csv.parent_path, outputs="image", color_flag='color'),
            ReadImage(inputs="mask_left",
                      parent_path=csv.parent_path,
                      outputs="mask_left",
                      color_flag='color',
                      mode='!infer'),
            ReadImage(inputs="mask_right",
                      parent_path=csv.parent_path,
                      outputs="mask_right",
                      color_flag='color',
                      mode='!infer'),
            CombineLeftRightMask(inputs=("mask_left", "mask_right"), outputs="mask", mode='!infer'),
            Resize(image_in="image", width=512, height=512),
            Resize(image_in="mask", width=512, height=512, mode='!infer'),
            Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="mask", mode='train')),
            Sometimes(numpy_op=Rotate(
                image_in="image", mask_in="mask", limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, mode='train')),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask", mode='!infer'),
            # use channel tranpose instead of the below line
            ChannelTranspose(inputs="image", outputs="image"),
            Reshape(shape=(3, 512, 512), inputs="mask", outputs="mask", mode='!infer'),
            # RepeatChannel(inputs="image", outputs="image") # Adds channels
        ])

    # step 2
    model = fe.build(model_fn=lambda: models.segmentation.deeplabv3_mobilenet_v3_large(),
                     optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=0.0001),
                     model_name="lung_segmentation")

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
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
