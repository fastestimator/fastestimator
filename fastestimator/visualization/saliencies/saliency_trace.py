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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from fastestimator.estimator.trace import Trace
from fastestimator.visualization.saliencies import plot_saliency


class Saliency(Trace):
    """Args:
        model_name (str): The model to be inspected by the Saliency visualization
        input_key (str): A string key corresponding to the tensor to be passed to the model
        baseline_constant (float): What constant value would a blank tensor have
        decode_dictionary (dict): A dictionary of "class_idx" -> "class_name" associations
        color_map (str): The color map to use to visualize the saliency maps.
                         Consider "Greys_r", "plasma", or "magma" as alternatives
        smooth (int): The number of samples to use when generating a smoothed image
    """
    def __init__(self,
                 model_name,
                 input_key,
                 n_inputs=1,
                 output_name=None,
                 im_freq=1,
                 baseline_constant=0,
                 decode_dictionary=None,
                 color_map="inferno",
                 smooth=7):

        self.data = []
        self.in_key = input_key
        if output_name is None:
            output_name = "{}_saliency".format(model_name)
        super().__init__(inputs=self.in_key, outputs=output_name, mode='eval')
        self.n_inputs = n_inputs
        self.output_key = output_name
        self.model_name = model_name
        self.model = None
        self.im_freq = im_freq
        self.baseline_constant = baseline_constant
        self.baseline = None
        self.decode_dictionary = decode_dictionary
        self.color_map = color_map
        self.smooth = smooth

    def on_begin(self, state):
        self.model = self.network.model[self.model_name]

    def on_batch_end(self, state):
        if state['epoch'] == 0 and len(self.data) <= self.n_inputs:
            self.data.append(state.get(self.in_key) or state['batch'][self.in_key])

    def on_epoch_end(self, state):
        if state['epoch'] % self.im_freq != 0:
            return
        if state['epoch'] == 0:
            self.data = tf.concat(self.data, axis=0)
            self.data = self.data[:self.n_inputs]
            self.baseline = tf.zeros_like(self.data, dtype=self.data.dtype) + self.baseline_constant
        fig = plot_saliency(self.model,
                            self.data,
                            baseline_input=self.baseline,
                            decode_dictionary=self.decode_dictionary,
                            color_map=self.color_map,
                            smooth=self.smooth)
        # TODO - Figure out how to get this to work without it displaying the figure. maybe fig.canvas.draw
        plt.draw()
        plt.pause(0.000001)
        state.maps[1][self.output_key] = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                                       sep='').reshape((1, ) + fig.canvas.get_width_height()[::-1] +
                                                                       (3, ))
        plt.close(fig)
