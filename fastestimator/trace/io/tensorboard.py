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
"""Trace contains metrics and other information users want to track."""
import datetime
import os

import tensorflow as tf
from tensorflow.python.framework import ops as tfops
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.ops import array_ops, summary_ops_v2

from fastestimator.trace import Trace
from fastestimator.util.util import is_number


class TensorBoard(Trace):
    """Output data for use in TensorBoard.

    Args:
        log_dir (str, optional): Path of the directory where to save the log files to be parsed by TensorBoard.
            Defaults to 'logs'.
        histogram_freq (int, optional): Frequency (in epochs) at which to compute activation and weight histograms for
            the layers of the model. If set to 0, histograms won't be computed. Defaults to 0.
        write_graph (bool, optional): Whether to visualize the graph in TensorBoard. The log file can become quite large
            when write_graph is set to True. Defaults to True.
        write_images (bool, str, list, optional): If True will write model weights to visualize as an image in
            TensorBoard. If a string or list of strings is provided, the corresponding keys will be written to
            Tensorboard images. To get weights and specific keys use [True, 'key1', 'key2',...] Defaults to False.
        update_freq (str, int, optional): 'batch' or 'epoch' or integer. When using 'batch', writes the losses and
            metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,
            the callback will write the metrics and losses to TensorBoard every 1000 samples. Note that writing too
            frequently to TensorBoard can slow down your training. Defaults to 'epoch'.
        profile_batch (int, optional): Which batch to run profiling on. 0 to disable. Note that FE batch numbering
            starts from 0 rather than 1. Defaults to 2.
        embeddings_freq (int, optional): Frequency (in epochs) at which embedding layers will be visualized. If set to
            0, embeddings won't be visualized.Defaults to 0.
        embeddings_metadata (str, dict, optional): A dictionary which maps layer name to a file name in which metadata
            for this embedding layer is saved. See the details about metadata files format. In case if the same
            metadata file is used for all embedding layers, string can be passed. Defaults to None.
    """
    def __init__(self,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None):
        super().__init__(inputs="*")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(os.path.join(log_dir, current_time), 'train')
        eval_log_dir = os.path.join(os.path.join(log_dir, current_time), 'eval')
        self.profile_log_dir = os.path.join(os.path.join(log_dir, current_time), 'profile')
        self.summary_writers = {
            'train': tf.summary.create_file_writer(self.train_log_dir),
            'eval': tf.summary.create_file_writer(eval_log_dir)
        }
        self.update_freq = 1 if update_freq == 'batch' else update_freq
        assert (self.update_freq == 'epoch' or (isinstance(self.update_freq, int) and self.update_freq > 0)), \
            "TensorBoard update_freq must be either 'epoch', 'batch', or a positive integer"
        self.ignore_keys = {'mode', 'epoch', 'train_step', 'batch_idx', 'batch_size', 'batch'}
        self.write_graph = write_graph
        self.write_images = {write_images} if isinstance(write_images, (str, bool)) else set(write_images)
        self.histogram_freq = histogram_freq
        self.profile_batch = profile_batch
        self.is_tracing = False
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata

    def on_begin(self, state):
        if self.write_graph:
            with self.summary_writers['train'].as_default():
                with summary_ops_v2.always_record_summaries():
                    summary_ops_v2.graph(backend.get_graph(), step=0)
                    for name, model in self.network.model.items():
                        summary_writable = (model._is_graph_network or model.__class__.__name__ == 'Sequential')
                        if summary_writable:
                            summary_ops_v2.keras_model(name, model, step=0)
        if self.embeddings_freq:
            self._configure_embeddings()

    def on_batch_end(self, state):
        if state['mode'] != 'train':
            return
        if self.is_tracing:
            self._log_trace(state['train_step'])
        elif state['train_step'] == self.profile_batch - 1:
            self._enable_trace()
        if self.update_freq == 'epoch' or state['train_step'] % self.update_freq != 0:
            return
        with self.summary_writers[state['mode']].as_default():
            for key in state.keys() - self.ignore_keys:
                val = state[key]
                if is_number(val):
                    tf.summary.scalar("batch_" + key, val, step=state['train_step'])
            for key in self.write_images - {True, False}:
                data = state.get(key)
                if data is not None:
                    tf.summary.image(key, data, step=state['train_step'])

    def on_epoch_end(self, state):
        with self.summary_writers[state['mode']].as_default():
            for key in state.keys() - self.ignore_keys:
                val = state[key]
                if is_number(val):
                    tf.summary.scalar("epoch_" + key, val, step=state['epoch'])
            for key in self.write_images - {True, False}:
                data = state.get(key)
                if data is not None:
                    tf.summary.image(key, data, step=state['epoch'])
            if state['mode'] == 'train' and self.histogram_freq and state['epoch'] % self.histogram_freq == 0:
                self._log_weights(epoch=state['epoch'])
            if state['mode'] == 'train' and self.embeddings_freq and state['epoch'] % self.embeddings_freq == 0:
                self._log_embeddings(state)

    def on_end(self, state):
        if self.is_tracing:
            self._log_trace(state['train_step'])
        for writer in self.summary_writers.values():
            writer.close()

    def _enable_trace(self):
        summary_ops_v2.trace_on(graph=True, profiler=True)
        self.is_tracing = True

    def _log_trace(self, global_batch_idx):
        with self.summary_writers['train'].as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.trace_export(name='batch_{}'.format(global_batch_idx),
                                        step=global_batch_idx,
                                        profiler_outdir=self.profile_log_dir)
        self.is_tracing = False

    def _log_embeddings(self, state):
        for model_name, model in self.network.model.items():
            embeddings_ckpt = os.path.join(self.train_log_dir,
                                           '{}_embedding.ckpt-{}'.format(model_name, state['epoch']))
            model.save_weights(embeddings_ckpt)

    def _log_weights(self, epoch):
        # Similar to TF implementation, but multiple models
        writer = self.summary_writers['train']
        with writer.as_default(), summary_ops_v2.always_record_summaries():
            for model_name, model in self.network.model.items():
                for layer in model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(':', '_')
                        weight_name = "{}_{}".format(model_name, weight_name)
                        with tfops.init_scope():
                            weight = backend.get_value(weight)
                        summary_ops_v2.histogram(weight_name, weight, step=epoch)
                        if True in self.write_images:
                            self._log_weight_as_image(weight, weight_name, epoch)
        writer.flush()

    @staticmethod
    def _log_weight_as_image(weight, weight_name, epoch):
        """ Logs a weight as a TensorBoard image.
            Implementation from tensorflow codebase, would have invoked theirs directly but they didn't make it a static
            method
        """
        w_img = array_ops.squeeze(weight)
        shape = backend.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = backend.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if backend.image_data_format() == 'channels_last':
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = backend.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])
        shape = backend.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            summary_ops_v2.image(weight_name, w_img, step=epoch)

    def _configure_embeddings(self):
        """Configure the Projector for embeddings.
        Implementation from tensorflow codebase, but supports multiple models
        """
        try:
            # noinspection PyPackageRequirements
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                              'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for model_name, model in self.network.model.items():
            for layer in model.layers:
                if isinstance(layer, embeddings.Embedding):
                    embedding = config.embeddings.add()
                    embedding.tensor_name = layer.embeddings.name

                    if self.embeddings_metadata is not None:
                        if isinstance(self.embeddings_metadata, str):
                            embedding.metadata_path = self.embeddings_metadata
                        else:
                            if layer.name in embedding.metadata_path:
                                embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

        if self.embeddings_metadata:
            raise ValueError('Unrecognized `Embedding` layer names passed to '
                             '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                             'argument: ' + str(self.embeddings_metadata))

        class DummyWriter(object):
            """Dummy writer to conform to `Projector` API."""
            def __init__(self, logdir):
                self.logdir = logdir

            def get_logdir(self):
                return self.logdir

        writer = DummyWriter(self.train_log_dir)
        projector.visualize_embeddings(writer, config)
