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
import matplotlib
import tensorflow as tf

from fastestimator.trace.io.xai import XAiTrace
from fastestimator.xai import UmapPlotter, fig_to_img


class UMap(XAiTrace):
    """
    Args:
        model_name (str): The model to be inspected by the visualization
        model_input (str, tf.Tensor): The input to the model, either a string key or the actual input tensor
        layer_id (int, str): Which layer to inspect. Defaults to the second-to-last layer
        n_inputs (int): How many inputs to be collected and passed to the model (if model_input is a string)
        resample_inputs (bool): Whether to re-sample inputs every im_freq iterations or use the same throughout training
                                Can only be True if model_input is a string
        output_key (str): The name of the output to be written into the batch dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        mode (str): The mode ('train', 'eval') on which to run the trace
        labels: The (optional) key of the classes corresponding to the inputs (used for coloring points)
        label_dictionary: An (optional) dictionary mapping labels from the label vector to other representations
                    (ex. {0:'dog', 1:'cat'})
        legend_loc: The location of the legend, or 'off' to disable figure legends
        **umap_parameters: Extra parameters to be passed to the umap algorithm, ex. n_neighbors, n_epochs, etc.
    """
    def __init__(self,
                 model_name,
                 model_input,
                 n_inputs=500,
                 resample_inputs=False,
                 layer_id=-2,
                 output_key=None,
                 im_freq=1,
                 mode="eval",
                 labels=None,
                 label_dictionary=None,
                 legend_loc='best',
                 **umap_parameters):

        super().__init__(model_name=model_name,
                         model_input=model_input,
                         n_inputs=n_inputs,
                         resample_inputs=resample_inputs,
                         output_key=output_key,
                         im_freq=im_freq,
                         mode=mode)

        if isinstance(labels, str):
            self.label_key = labels
            self.inputs.add(self.label_key)
            self.labels = {"train": [], "eval": []}
        else:
            self.label_key = None
            self.labels = labels
        self.umap = UmapPlotter(label_dict=label_dictionary, **umap_parameters)
        self.legend_loc = legend_loc
        self.layer_id = layer_id

    def on_begin(self, state):
        super().on_begin(state)
        if isinstance(self.layer_id, int):
            self.layer_id = self.model.layers[self.layer_id].name
        self.model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.layer_id).output)

    def on_epoch_begin(self, state):
        super().on_epoch_begin(state)
        if self.resample_inputs and self._data_epoch(state['epoch']):
            self.labels[state['mode']] = []

    def on_batch_end(self, state):
        if self._data_epoch(state['epoch']) and self.collected_inputs[state['mode']] < self.n_inputs and self.label_key:
            samples = state.get(self.label_key) or state["batch"][self.label_key]
            self.labels[state['mode']].append(samples)
        super().on_batch_end(state)

    def on_epoch_end(self, state):
        super().on_epoch_end(state)
        if self._data_epoch(state['epoch']) and self.label_key:
            self.labels[state['mode']] = tf.concat(self.labels[state['mode']], axis=0)
            self.labels[state['mode']] = self.labels[state['mode']][:self.n_inputs]
        if state['epoch'] % self.im_freq != 0:
            return
        model_output = self.model(self.data[state['mode']])
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        fig = self.umap.plot_umap(model_output,
                                  labels=self.labels[state['mode']] if self.labels else None,
                                  legend_loc=self.legend_loc)
        fig.canvas.draw()
        state.maps[1][self.output_key] = fig_to_img(fig)
        matplotlib.use(old_backend)
