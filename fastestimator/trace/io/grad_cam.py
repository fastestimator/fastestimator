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
import cv2
import matplotlib

from fastestimator.xai import plot_gradcam, fig_to_img
from fastestimator.trace.io.xai import XAiTrace


class GradCam(XAiTrace):
    """
    Draw GradCam heatmaps for given inputs

    Args:
        model_name (str): The model to be inspected by the visualization
        model_input (str, tf.Tensor): The input to the model, either a string key or the actual input tensor
        n_inputs (int): How many inputs to be collected and passed to the model (if model_input is a string)
        resample_inputs (bool): Whether to re-sample inputs every im_freq iterations or use the same throughout training
                                Can only be True if model_input is a string
        output_key (str): The name of the output to be written into the batch dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        mode (str): The mode ('train', 'eval') on which to run the trace
        layer_id (int, str, None): Which layer to inspect. Should be a convolutional layer. If None, the last \
                                    acceptable layer from the model will be selected
        label_dictionary (dict): A dictionary of "class_idx" -> "class_name" associations
        color_map (int): Which colormap to use when generating the heatmaps
    """
    def __init__(self,
                 model_name,
                 model_input,
                 n_inputs=1,
                 resample_inputs=False,
                 layer_id=None,
                 output_key=None,
                 im_freq=1,
                 mode="eval",
                 label_dictionary=None,
                 color_map=cv2.COLORMAP_INFERNO):

        super().__init__(model_name=model_name,
                         model_input=model_input,
                         n_inputs=n_inputs,
                         resample_inputs=resample_inputs,
                         output_key=output_key,
                         im_freq=im_freq,
                         mode=mode)

        self.color_map = color_map
        self.layer_id = layer_id
        self.label_dictionary = label_dictionary

    def on_begin(self, state):
        super().on_begin(state)
        if isinstance(self.layer_id, int):
            self.layer_id = self.model.layers[self.layer_id].name
        if self.layer_id is None:
            for layer in reversed(self.model.layers):
                if layer.output.shape.ndims == 4:
                    self.layer_id = layer.name
                    break
        assert self.model.get_layer(self.layer_id).output.shape.ndims == 4, \
            "GradCam must run on a layer with outputs of the form [batch, height, width, filters] (a convolution layer)"

    def on_epoch_end(self, state):
        super().on_epoch_end(state)
        if state['epoch'] % self.im_freq != 0:
            return
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        fig = plot_gradcam(self.data[state['mode']],
                           self.model,
                           self.layer_id,
                           decode_dictionary=self.label_dictionary,
                           colormap=self.color_map)
        fig.canvas.draw()
        state.maps[1][self.output_key] = fig_to_img(fig)
        matplotlib.use(old_backend)
