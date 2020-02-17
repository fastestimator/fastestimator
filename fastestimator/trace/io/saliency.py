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
from fastestimator.xai import plot_saliency, fig_to_img


class Saliency(XAiTrace):
    """
    Args:
        model_name (str): The model to be inspected by the Saliency visualization
        model_input (str, tf.Tensor): The input to the model, either a string key or the actual input tensor
        n_inputs (int): How many samples should be drawn from the input_key tensor for visualization
        resample_inputs (bool): Whether to re-sample inputs every im_freq iterations or use the same throughout training
                                Can only be True if model_input is a string
        output_key (str): The name of the output to be written into the batch dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        mode (str): The mode ('train', 'eval') on which to run the trace
        label_dictionary (dict): A dictionary of "class_idx" -> "class_name" associations
        baseline_constant (float): What constant value would a blank tensor have
        color_map (str): The color map to use to visualize the saliency maps.
                         Consider "Greys_r", "plasma", or "magma" as alternatives
        smooth (int): The number of samples to use when generating a smoothed image
    """
    def __init__(self,
                 model_name,
                 model_input,
                 n_inputs=1,
                 resample_inputs=False,
                 output_key=None,
                 im_freq=1,
                 mode='eval',
                 label_dictionary=None,
                 baseline_constant=0,
                 color_map="inferno",
                 smooth=7):

        super().__init__(model_name=model_name,
                         model_input=model_input,
                         n_inputs=n_inputs,
                         resample_inputs=resample_inputs,
                         output_key=output_key,
                         im_freq=im_freq,
                         mode=mode)

        self.baseline_constant = baseline_constant
        self.baseline = None
        self.label_dictionary = label_dictionary
        self.color_map = color_map
        self.smooth = smooth

    def on_epoch_end(self, state):
        super().on_epoch_end(state)
        if self._data_epoch(state['epoch']):
            self.baseline = tf.zeros_like(self.data, dtype=self.data.dtype) + self.baseline_constant
        if state['epoch'] % self.im_freq != 0:
            return
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        fig = plot_saliency(self.model,
                            self.data[state['mode']],
                            baseline_input=self.baseline,
                            decode_dictionary=self.label_dictionary,
                            color_map=self.color_map,
                            smooth=self.smooth)
        fig.canvas.draw()
        state.maps[1][self.output_key] = fig_to_img(fig)
        matplotlib.use(old_backend)
