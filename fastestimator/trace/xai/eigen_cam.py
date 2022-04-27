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
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Iterable, Union

import cv2
import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._argmax import argmax
from fastestimator.backend._concat import concat
from fastestimator.backend._get_image_dims import get_image_dims
from fastestimator.backend._reduce_max import reduce_max
from fastestimator.backend._squeeze import squeeze
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.img_data import GridDisplay, BatchDisplay
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class EigenCAM(Trace):
    """A trace which draws EigenCAM heatmaps on top of images.

    These are useful for visualizing the outputs of the feature extractor component of a model. They are relatively
    insensitive to adversarial attacks, so don't use them to try and detect those. See https://arxiv.org/abs/2008.00299
    for more details.

    Args:
        images: The key corresponding to images onto which to draw the CAM outputs.
        activations: The key corresponding to outputs from a convolution layer from which to draw the CAM outputs. You
            can easily extract these from any model by using the 'intermediate_layers' variable in a ModelOp.
        n_components: How many principal components to visualize. If you pass a float between 0 and 1 it will instead
            visualize however many components are required in order to capture the corresponding percentage of the
            variance in the image.
        n_samples: How many images in total to display every epoch (or None to display all available images).
        downsize: Whether to downsize the inputs before svd decomposition in order to speed up processing. If provided,
            the inputs will be proportionally downscaled such that their longest axis length is equal to the `downsize`
            parameter. 64 seems like a good value to try if you are having performance problems.
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
                 activations: str,
                 n_components: Union[int, float] = 3,
                 n_samples: Optional[int] = 5,
                 downsize: Optional[int] = None,
                 labels: Optional[str] = None,
                 preds: Optional[str] = None,
                 label_mapping: Optional[Dict[str, Any]] = None,
                 outputs: str = "eigencam",
                 mode: Union[None, str, Iterable[str]] = "!train",
                 ds_id: Union[None, str, Iterable[str]] = None):
        self.image_key = images
        self.activation_key = activations
        self.true_label_key = labels
        self.pred_label_key = preds
        inputs = [x for x in (images, activations, labels, preds) if x is not None]
        self.n_components = n_components
        self.n_samples = n_samples
        # TODO - handle non-hashable labels
        self.label_mapping = {val: key for key, val in label_mapping.items()} if label_mapping else None
        self.downsize = downsize
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.images = []
        self.activations = []
        self.labels = []
        self.preds = []
        self.n_found = 0

    def _reset(self) -> None:
        """Clear memory for next epoch.
        """
        self.images = []
        self.activations = []
        self.labels = []
        self.preds = []
        self.n_found = 0

    def _project_2d(self, activations: np.ndarray) -> Tuple[int, List[List[np.ndarray]]]:
        """Project 2D convolution activations maps into 2D principal component maps.

        Args:
            activations: A tensor of shape (batch, channels, height, width) to be transformed.

        Returns:
            (max N_components, Principal component projections of the `activations` (batch x components x image)).
        """
        projections = []
        for activation in activations:
            if self.downsize:
                long_axis = 1 if activation.shape[1] > activation.shape[2] else 2
                long_len = activation.shape[long_axis]
                if long_len > self.downsize:
                    scale = self.downsize / long_len
                    small_activations = []
                    for i in range(activation.shape[0]):
                        small_activations.append(
                            cv2.resize(src=activation[i, ...],
                                       dsize=(int(activation.shape[1]*scale), int(activation.shape[2]*scale)),
                                       interpolation=cv2.INTER_AREA))
                    activation = np.array(small_activations)
            flat = activation.reshape(activation.shape[0], -1).transpose()
            flat = flat - flat.mean(axis=0)
            U, S, VT = np.linalg.svd(flat, full_matrices=True)
            components = []
            n_components = self.n_components
            if isinstance(n_components, float):
                eig_vals = np.square(S)
                pct_explained = np.cumsum(eig_vals) / np.cumsum(eig_vals)[-1]
                n_components = 1 + np.searchsorted(pct_explained, self.n_components)
            for i in range(n_components):
                component_i = flat @ VT[i, :]
                component_i = component_i.reshape(activation.shape[1:])
                components.append(np.maximum(component_i, 0))
            projections.append(components)
        return max([len(x) for x in projections]), projections

    def on_batch_end(self, data: Data) -> None:
        if self.n_samples is None or self.n_found < self.n_samples:
            self.images.append(data[self.image_key])
            self.activations.append(data[self.activation_key])
            if self.true_label_key:
                self.labels.append(data[self.true_label_key])
            if self.pred_label_key:
                self.preds.append(data[self.pred_label_key])
            self.n_found += len(data[self.image_key])

    def on_epoch_end(self, data: Data) -> None:
        # Keep only the user-specified number of samples
        images = concat(self.images)[:self.n_samples or self.n_found]
        _, height, width = get_image_dims(images)
        activations = to_number(concat(self.activations)[:self.n_samples or self.n_found])
        if tf.is_tensor(images):
            activations = np.moveaxis(activations, source=-1, destination=1)  # Activations should be channel first
        columns = []
        labels = None if not self.labels else concat(self.labels)[:self.n_samples or self.n_found]
        if labels is not None:
            if len(labels.shape) > 1:
                labels = argmax(labels, axis=-1)
            if self.label_mapping:
                labels = np.array([self.label_mapping[clazz] for clazz in to_number(squeeze(labels))])
            columns.append(BatchDisplay(text=labels, title=self.true_label_key))
        preds = None if not self.preds else concat(self.preds)[:self.n_samples or self.n_found]
        if preds is not None:
            if len(preds.shape) > 1:
                preds = argmax(preds, axis=-1)
            if self.label_mapping:
                preds = np.array([self.label_mapping[clazz] for clazz in to_number(squeeze(preds))])
            columns.append(BatchDisplay(text=preds, title=self.pred_label_key))
        columns.append(BatchDisplay(image=images, title=self.image_key))
        # Clear memory
        self._reset()
        # Make the image
        n_components, batch_component_image = self._project_2d(activations)
        components = []  # component x image (batch x image)
        for component_idx in range(n_components):
            batch = []
            for base_image, component_image in zip(images, batch_component_image):
                if len(component_image) > component_idx:
                    mask = component_image[component_idx]
                    mask = cv2.resize(mask, (width, height))
                    mask = mask - np.min(mask)
                    mask = mask / np.max(mask)
                    mask = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    mask = np.float32(mask) / 255
                    # switch to channel first for pytorch
                    if isinstance(base_image, torch.Tensor):
                        mask = np.moveaxis(mask, source=-1, destination=1)
                    new_image = base_image + mask
                    new_image = new_image / reduce_max(new_image)
                else:
                    # There's no component for this image, so display an empty image here
                    new_image = np.ones_like(base_image)
                batch.append(new_image)
            components.append(np.array(batch, dtype=np.float32))

        for idx, elem in enumerate(components):
            columns.append(BatchDisplay(image=elem, title=f"Component {idx}"))

        result = GridDisplay(columns=columns)
        data.write_without_log(self.outputs[0], result)
