#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
from typing import Any, Dict, Iterable, Optional, TypeVar, Union

import cv2
import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.argmax import argmax
from fastestimator.backend.concat import concat
from fastestimator.backend.get_image_dims import get_image_dims
from fastestimator.backend.reduce_max import reduce_max
from fastestimator.backend.squeeze import squeeze
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.img_data import ImgData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class GradCAM(Trace):
    """A trace which draws GradCAM heatmaps on top of images.

    These are useful for visualizing supports for a model's classification. See https://arxiv.org/pdf/1610.02391.pdf
    for more details.

    Args:
        images: The key corresponding to images onto which to draw the CAM outputs.
        grads: The key corresponding to gradients of the model output with respect to a convolution layer of the model.
            You can easily extract these from any model by using the 'intermediate_layers' variable in a ModelOp, along
            with the GradientOp. Make sure to select a particular component of y_pred when computing gradients rather
            than using the entire vector. See our GradCAM XAI tutorial for an example.
        n_components: How many principal components to visualize.
        n_samples: How many images in total to display every epoch (or None to display all available images).
        labels: The key corresponding to the true labels of the images to be visualized.
        preds: The key corresponding to the model prediction for each image.
        label_mapping: {class_string: model_output_value}.
        outputs: The key into which to write the eigencam images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 images: str,
                 grads: str,
                 n_components: int = 3,
                 n_samples: Optional[int] = 5,
                 labels: Optional[str] = None,
                 preds: Optional[str] = None,
                 label_mapping: Optional[Dict[str, Any]] = None,
                 outputs: str = "gradcam",
                 mode: Union[None, str, Iterable[str]] = "!train",
                 ds_id: Union[None, str, Iterable[str]] = None):
        self.image_key = images
        self.grad_key = grads
        self.true_label_key = labels
        self.pred_label_key = preds
        inputs = [x for x in (images, grads, labels, preds) if x is not None]
        self.n_components = n_components
        self.n_samples = n_samples
        # TODO - handle non-hashable labels
        self.label_mapping = {val: key for key, val in label_mapping.items()} if label_mapping else None
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.images = []
        self.grads = []
        self.labels = []
        self.preds = []
        self.n_found = 0

    def _reset(self) -> None:
        """Clear memory for next epoch.
        """
        self.images = []
        self.grads = []
        self.labels = []
        self.preds = []
        self.n_found = 0

    def on_batch_end(self, data: Data) -> None:
        if self.n_samples is None or self.n_found < self.n_samples:
            self.images.append(data[self.image_key])
            self.grads.append(data[self.grad_key])
            if self.true_label_key:
                self.labels.append(data[self.true_label_key])
            if self.pred_label_key:
                self.preds.append(data[self.pred_label_key])
            self.n_found += len(data[self.image_key])

    def on_epoch_end(self, data: Data) -> None:
        # Keep only the user-specified number of samples
        images = concat(self.images)[:self.n_samples or self.n_found]
        _, height, width = get_image_dims(images)
        grads = to_number(concat(self.grads)[:self.n_samples or self.n_found])
        if tf.is_tensor(images):
            grads = np.moveaxis(grads, source=-1, destination=1)  # grads should be channel first
        args = {}
        labels = None if not self.labels else concat(self.labels)[:self.n_samples or self.n_found]
        if labels is not None:
            if len(labels.shape) > 1:
                labels = argmax(labels, axis=-1)
            if self.label_mapping:
                labels = np.array([self.label_mapping[clazz] for clazz in to_number(squeeze(labels))])
            args[self.true_label_key] = labels
        preds = None if not self.preds else concat(self.preds)[:self.n_samples or self.n_found]
        if preds is not None:
            if len(preds.shape) > 1:
                preds = argmax(preds, axis=-1)
            if self.label_mapping:
                preds = np.array([self.label_mapping[clazz] for clazz in to_number(squeeze(preds))])
            args[self.pred_label_key] = preds
        args[self.image_key] = images
        # Clear memory
        self._reset()
        # Make the image
        # TODO: In future maybe allow multiple different grads to have side-by-side comparisons of classes
        components = [np.mean(grads, axis=1)]
        components = [np.maximum(component, 0) for component in components]
        masks = []
        for component_batch in components:
            img_batch = []
            for img in component_batch:
                img = cv2.resize(img, (height, width))
                img = img - np.min(img)
                img = img / np.max(img)
                img = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                img = np.float32(img) / 255
                img_batch.append(img)
            img_batch = np.array(img_batch, dtype=np.float32)
            # Switch to channel first for pytorch
            if isinstance(images, torch.Tensor):
                img_batch = np.moveaxis(img_batch, source=-1, destination=1)
            masks.append(img_batch)

        components = [images + mask for mask in masks]  # This seems to work even if the image is 1 channel instead of 3
        components = [image / reduce_max(image) for image in components]

        for elem in components:
            args[self.grad_key] = elem

        result = ImgData(**args)
        data.write_without_log(self.outputs[0], result)
