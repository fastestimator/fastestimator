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
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import tensorboard as tb
import tensorflow as tf
import torch
from plotly.graph_objs import Figure
from tensorflow.keras import backend
from tensorflow.python.framework import ops as tfops
from tensorflow.python.keras.callbacks import keras_model_summary
from tensorflow.python.ops import summary_ops_v2
from torch.utils.tensorboard import SummaryWriter

from fastestimator.backend._abs import abs
from fastestimator.backend._concat import concat
from fastestimator.backend._expand_dims import expand_dims
from fastestimator.backend._permute import permute
from fastestimator.backend._reduce_sum import reduce_sum
from fastestimator.backend._reshape import reshape
from fastestimator.backend._squeeze import squeeze
from fastestimator.backend._to_tensor import to_tensor
from fastestimator.network import BaseNetwork, TFNetwork
from fastestimator.trace.trace import Trace, parse_freq
from fastestimator.util.base_util import DefaultKeyDict, is_number, to_list, to_set
from fastestimator.util.data import Data
from fastestimator.util.img_data import Display
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number

# https://github.com/pytorch/pytorch/issues/30966
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)
Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class _BaseWriter:
    """A class to write various types of data into TensorBoard summary files.

    This class is intentionally not @traceable.

    Args:
        root_log_dir: The directory into which to store a new directory corresponding to this experiment's summary data
        time_stamp: The timestamp of this experiment (used as a folder name within `root_log_dir`).
        network: The network associated with the current experiment.
    """
    summary_writers: Dict[str, SummaryWriter]
    network: BaseNetwork

    def __init__(self, root_log_dir: str, time_stamp: str, network: BaseNetwork) -> None:
        self.summary_writers = DefaultKeyDict(lambda key:
                                              (SummaryWriter(log_dir=os.path.join(root_log_dir, time_stamp, key))))
        self.network = network

    def write_epoch_models(self, mode: str, epoch: int) -> None:
        """Write summary graphs for all of the models in the current epoch.

        Args:
            mode: The current mode of execution ('train', 'eval', 'test', 'infer').
            epoch: The current epoch of execution.
        """
        raise NotImplementedError

    def write_weights(self, mode: str, models: Iterable[Model], step: int, visualize: bool) -> None:
        """Write summaries of all of the weights of a given collection of `models`.

        Args:
            mode: The current mode of execution ('train', 'eval', 'test', 'infer').
            models: A list of models compiled with fe.build whose weights should be recorded.
            step: The current training step.
            visualize: Whether to attempt to paint graphical representations of the weights in addition to the default
                histogram summaries.
        """
        raise NotImplementedError

    def write_scalars(self, mode: str, scalars: Iterable[Tuple[str, Any]], step: int) -> None:
        """Write summaries of scalars to TensorBoard.

        Args:
            mode: The current mode of execution ('train', 'eval', 'test', 'infer').
            scalars: A collection of pairs like [("key", val), ("key2", val2), ...].
            step: The current training step.
        """
        for key, val in scalars:
            self.summary_writers[mode].add_scalar(tag=key, scalar_value=to_number(val), global_step=step)

    def write_images(self, mode: str, images: Iterable[Tuple[str, Any]], step: int) -> None:
        """Write images to TensorBoard.

        Args:
            mode: The current mode of execution ('train', 'eval', 'test', 'infer').
            images: A collection of pairs like [("key", image1), ("key2", image2), ...].
            step: The current training step.
        """
        for key, img in images:
            if isinstance(img, Display):
                img = img.prepare()
            if isinstance(img, Figure):
                img = img.to_image(format='png')
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.summary_writers[mode].add_image(tag=key, img_tensor=img, global_step=step, dataformats='HWC')
            else:
                self.summary_writers[mode].add_images(tag=key,
                                                      img_tensor=to_number(img),
                                                      global_step=step,
                                                      dataformats='NCHW' if isinstance(img, torch.Tensor) else 'NHWC')

    def write_embeddings(
            self,
            mode: str,
            embeddings: Iterable[Tuple[str, Tensor, Optional[List[Any]], Optional[Tensor]]],
            step: int,
    ):
        """Write embeddings (like UMAP) to TensorBoard.

        Args:
            mode: The current mode of execution ('train', 'eval', 'test', 'infer').
            embeddings: A collection of quadruplets like [("key", <features>, [<label1>, ...], <label_images>)].
                Features are expected to be batched, and if labels and/or label images are provided they should have the
                same batch dimension as the features.
            step: The current training step.
        """
        for key, features, labels, label_imgs in embeddings:
            flat = to_number(reshape(features, [features.shape[0], -1]))
            if not isinstance(label_imgs, (torch.Tensor, type(None))):
                label_imgs = to_tensor(label_imgs, 'torch')
                if len(label_imgs.shape) == 4:
                    label_imgs = permute(label_imgs, [0, 3, 1, 2])
            self.summary_writers[mode].add_embedding(mat=flat,
                                                     metadata=labels,
                                                     label_img=label_imgs,
                                                     tag=key,
                                                     global_step=step)

    def close(self) -> None:
        """A method to flush and close all connections to the files on disk.
        """
        modes = list(self.summary_writers.keys())  # break connection with dictionary so can delete in iteration
        for mode in modes:
            self.summary_writers[mode].close()
            del self.summary_writers[mode]

    @staticmethod
    def _weight_to_image(weight: Tensor, kernel_channels_last: bool = False) -> Optional[Tensor]:
        """Logs a weight as a TensorBoard image.

        Implementation from TensorFlow codebase, would have invoked theirs directly but they didn't make it a static
        method.
        """
        w_img = squeeze(weight)
        shape = backend.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = permute(w_img, [0, 1])
                shape = backend.int_shape(w_img)
            w_img = reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if kernel_channels_last:
                # Switch to channels_first to display every kernel as a separate images
                w_img = permute(w_img, [2, 0, 1])
            w_img = expand_dims(w_img, axis=-1)
        elif len(shape) == 4:  # Conv filter with multiple input channels
            if kernel_channels_last:
                # Switch to channels first to display kernels as separate images
                w_img = permute(w_img, [3, 2, 0, 1])
            w_img = reduce_sum(abs(w_img), axis=1)  # Sum over the each channel within the kernel
            w_img = expand_dims(w_img, axis=-1)
        shape = backend.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            return w_img


class _TfWriter(_BaseWriter):
    """A class to write various TensorFlow data into TensorBoard summary files.

    This class is intentionally not @traceable.

    Args:
        root_log_dir: The directory into which to store a new directory corresponding to this experiment's summary data
        time_stamp: The timestamp of this experiment (used as a folder name within `root_log_dir`).
        network: The network associated with the current experiment.
    """
    tf_summary_writers: Dict[str, tf.summary.SummaryWriter]

    def __init__(self, root_log_dir: str, time_stamp: str, network: TFNetwork) -> None:
        super().__init__(root_log_dir=root_log_dir, time_stamp=time_stamp, network=network)
        self.tf_summary_writers = DefaultKeyDict(
            lambda key: (tf.summary.create_file_writer(os.path.join(root_log_dir, time_stamp, key))))

    def write_epoch_models(self, mode: str, epoch: int) -> None:
        with self.tf_summary_writers[mode].as_default(), summary_ops_v2.always_record_summaries():
            # Record the overall execution summary
            if hasattr(self.network._forward_step_static, '_concrete_stateful_fn'):
                # noinspection PyProtectedMember
                summary_ops_v2.graph(self.network._forward_step_static._concrete_stateful_fn.graph)
            # Record the individual model summaries
            for model in self.network.epoch_models:
                summary_writable = (model.__class__.__name__ == 'Sequential'
                                    or (hasattr(model, '_is_graph_network') and model._is_graph_network))
                if summary_writable:
                    keras_model_summary(model.model_name, model, step=epoch)

    def write_weights(self, mode: str, models: Iterable[Model], step: int, visualize: bool) -> None:
        # Similar to TF implementation, but multiple models
        with self.tf_summary_writers[mode].as_default(), summary_ops_v2.always_record_summaries():
            for model in models:
                for layer in model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(':', '_')
                        weight_name = "{}_{}".format(model.model_name, weight_name)
                        with tfops.init_scope():
                            weight = backend.get_value(weight)
                        summary_ops_v2.histogram(weight_name, weight, step=step)
                        if visualize:
                            weight = self._weight_to_image(weight=weight, kernel_channels_last=True)
                            if weight is not None:
                                summary_ops_v2.image(weight_name, weight, step=step, max_images=weight.shape[0])

    def close(self) -> None:
        super().close()
        modes = list(self.tf_summary_writers.keys())  # break connection with dictionary so can delete in iteration
        for mode in modes:
            self.tf_summary_writers[mode].close()
            del self.tf_summary_writers[mode]


class _TorchWriter(_BaseWriter):
    """A class to write various Pytorch data into TensorBoard summary files.

    This class is intentionally not @traceable.
    """
    def write_epoch_models(self, mode: str, epoch: int) -> None:
        for model in self.network.epoch_models:
            inputs = model.fe_input_spec.get_dummy_input()
            self.summary_writers[mode].add_graph(model.module if torch.cuda.device_count() > 1 else model,
                                                 input_to_model=inputs)

    def write_weights(self, mode: str, models: Iterable[Model], step: int, visualize: bool) -> None:
        for model in models:
            for name, params in model.named_parameters():
                name = name.replace(".", "/")
                name = "{}_{}".format(model.model_name, name)
                weight = params.data
                self.summary_writers[mode].add_histogram(tag=name, values=weight, global_step=step)
                if visualize:
                    weight = self._weight_to_image(weight=weight)
                    if weight is not None:
                        self.summary_writers[mode].add_images(tag=name + "/image",
                                                              img_tensor=weight,
                                                              global_step=step,
                                                              dataformats='NHWC')


@traceable()
class TensorBoard(Trace):
    """Output data for use in TensorBoard.

    Note that if you plan to run a tensorboard server simultaneous to training, you may want to consider using the
    --reload_multifile=true flag until their multi-writer use case is finished:
    https://github.com/tensorflow/tensorboard/issues/1063

    Args:
        log_dir: Path of the directory where the log files to be parsed by TensorBoard should be saved.
        update_freq: 'batch', 'epoch', integer, or strings like '10s', '15e'. When using 'batch', writes the losses and
            metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,
            the callback will write the metrics and losses to TensorBoard every 1000 samples. You can also use strings
            like '8s' to indicate every 8 steps or '5e' to indicate every 5 epochs. Note that writing too frequently to
            TensorBoard can slow down your training. You can use None to disable updating, but this will make the trace
            mostly useless.
        write_graph: Whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph
            is set to True.
        write_images: If a string or list of strings is provided, the corresponding keys will be written to TensorBoard
            images.
        weight_histogram_freq: Frequency (in epochs) at which to compute activation and weight histograms for the layers
            of the model. Same argument format as `update_freq`.
        paint_weights: If True the system will attempt to visualize model weights as an image.
        write_embeddings: If a string or list of strings is provided, the corresponding keys will be written to
            TensorBoard embeddings.
        embedding_labels: Keys corresponding to label information for the `write_embeddings`.
        embedding_images: Keys corresponding to raw images to be associated with the `write_embeddings`.
    """
    writer: _BaseWriter

    # TODO - support for per-instance tracking

    def __init__(self,
                 log_dir: str = 'logs',
                 update_freq: Union[None, int, str] = 100,
                 write_graph: bool = True,
                 write_images: Union[None, str, List[str]] = None,
                 weight_histogram_freq: Union[None, int, str] = None,
                 paint_weights: bool = False,
                 embedding_freq: Union[None, int, str] = 'epoch',
                 write_embeddings: Union[None, str, List[str]] = None,
                 embedding_labels: Union[None, str, List[str]] = None,
                 embedding_images: Union[None, str, List[str]] = None) -> None:
        super().__init__(inputs=["*"] + to_list(write_images) + to_list(write_embeddings) + to_list(embedding_labels) +
                                to_list(embedding_images))
        self.root_log_dir = log_dir
        self.update_freq = parse_freq(update_freq)
        self.write_graph = write_graph
        self.painted_graphs = set()
        self.write_images = to_set(write_images)
        self.histogram_freq = parse_freq(weight_histogram_freq)
        if paint_weights and self.histogram_freq.freq == 0:
            self.histogram_freq.is_step = False
            self.histogram_freq.freq = 1
        self.paint_weights = paint_weights
        if write_embeddings is None and embedding_labels is None and embedding_images is None:
            # Speed up if-check short-circuiting later
            embedding_freq = None
        self.embedding_freq = parse_freq(embedding_freq)
        write_embeddings = to_list(write_embeddings)
        embedding_labels = to_list(embedding_labels)
        if embedding_labels:
            assert len(embedding_labels) == len(write_embeddings), \
                f"Expected {len(write_embeddings)} embedding_labels keys, but recieved {len(embedding_labels)}. Use \
                None to pad out the list if you have labels for only a subset of all embeddings."

        else:
            embedding_labels = [None for _ in range(len(write_embeddings))]
        embedding_images = to_list(embedding_images)
        if embedding_images:
            assert len(embedding_images) == len(write_embeddings), \
                f"Expected {len(write_embeddings)} embedding_images keys, but recieved {len(embedding_images)}. Use \
                None to pad out the list if you have labels for only a subset of all embeddings."

        else:
            embedding_images = [None for _ in range(len(write_embeddings))]
        self.write_embeddings = [(feature, label, img_label) for feature,
                                                                 label,
                                                                 img_label in
                                 zip(write_embeddings, embedding_labels, embedding_images)]
        self.collected_embeddings = defaultdict(list)

    def on_begin(self, data: Data) -> None:
        print("FastEstimator-Tensorboard: writing logs to {}".format(
            os.path.abspath(os.path.join(self.root_log_dir, self.system.experiment_time))))
        self.writer = _TfWriter(self.root_log_dir, self.system.experiment_time, self.system.network) if isinstance(
            self.system.network, TFNetwork) else _TorchWriter(
            self.root_log_dir, self.system.experiment_time, self.system.network)
        if self.write_graph and self.system.global_step == 1:
            self.painted_graphs = set()

    def on_batch_end(self, data: Data) -> None:
        if self.write_graph and self.system.network.epoch_models.symmetric_difference(self.painted_graphs):
            self.writer.write_epoch_models(mode=self.system.mode, epoch=self.system.epoch_idx)
            self.painted_graphs = self.system.network.epoch_models
        # Collect embeddings if present in batch but viewing per epoch. Don't aggregate during training though
        if self.system.mode != 'train' and self.embedding_freq.freq and not self.embedding_freq.is_step and \
                self.system.epoch_idx % self.embedding_freq.freq == 0:
            for elem in self.write_embeddings:
                name, lbl, img = elem
                if name in data:
                    self.collected_embeddings[name].append((data.get(name), data.get(lbl), data.get(img)))
        # Handle embeddings if viewing per step
        if self.embedding_freq.freq and self.embedding_freq.is_step and \
                self.system.global_step % self.embedding_freq.freq == 0:
            self.writer.write_embeddings(
                mode=self.system.mode,
                step=self.system.global_step,
                embeddings=filter(
                    lambda x: x[1] is not None,
                    map(lambda t: (t[0], data.get(t[0]), data.get(t[1]), data.get(t[2])), self.write_embeddings)))
        if self.system.mode != 'train':
            return
        if self.histogram_freq.freq and self.histogram_freq.is_step and \
                self.system.global_step % self.histogram_freq.freq == 0:
            self.writer.write_weights(mode=self.system.mode,
                                      models=self.system.network.models,
                                      step=self.system.global_step,
                                      visualize=self.paint_weights)
        if self.update_freq.freq and self.update_freq.is_step and self.system.global_step % self.update_freq.freq == 0:
            self.writer.write_scalars(mode=self.system.mode,
                                      step=self.system.global_step,
                                      scalars=filter(lambda x: is_number(x[1]), data.items()))
            self.writer.write_images(
                mode=self.system.mode,
                step=self.system.global_step,
                images=filter(lambda x: x[1] is not None, map(lambda y: (y, data.get(y)), self.write_images)))

    def on_epoch_end(self, data: Data) -> None:
        if self.system.mode == 'train' and self.histogram_freq.freq and not self.histogram_freq.is_step and \
                self.system.epoch_idx % self.histogram_freq.freq == 0:
            self.writer.write_weights(mode=self.system.mode,
                                      models=self.system.network.models,
                                      step=self.system.global_step,
                                      visualize=self.paint_weights)
        # Write out any embeddings which were aggregated over batches
        for name, val_list in self.collected_embeddings.items():
            embeddings = None if any(x[0] is None for x in val_list) else concat([x[0] for x in val_list])
            labels = None if any(x[1] is None for x in val_list) else concat([x[1] for x in val_list])
            imgs = None if any(x[2] is None for x in val_list) else concat([x[2] for x in val_list])
            self.writer.write_embeddings(mode=self.system.mode,
                                         step=self.system.global_step,
                                         embeddings=[(name, embeddings, labels, imgs)])
        self.collected_embeddings.clear()
        # Get any embeddings which were generated externally on epoch end
        if self.embedding_freq.freq and (self.embedding_freq.is_step
                                         or self.system.epoch_idx % self.embedding_freq.freq == 0):
            self.writer.write_embeddings(
                mode=self.system.mode,
                step=self.system.global_step,
                embeddings=filter(
                    lambda x: x[1] is not None,
                    map(lambda t: (t[0], data.get(t[0]), data.get(t[1]), data.get(t[2])), self.write_embeddings)))
        if self.update_freq.freq and (self.update_freq.is_step or self.system.epoch_idx % self.update_freq.freq == 0):
            self.writer.write_scalars(mode=self.system.mode,
                                      step=self.system.global_step,
                                      scalars=filter(lambda x: is_number(x[1]), data.items()))
            self.writer.write_images(
                mode=self.system.mode,
                step=self.system.global_step,
                images=filter(lambda x: x[1] is not None, map(lambda y: (y, data.get(y)), self.write_images)))

    def on_end(self, data: Data) -> None:
        self.writer.close()
