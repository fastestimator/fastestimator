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

from fastestimator.trace import Trace
from fastestimator.interpretation import plot_caricature
from fastestimator.util.util import to_list, Suppressor


class Caricature(Trace):
    """
    Args:
        model_name (str): The model to be inspected by the Caricature visualization
        layer_ids (int, list): The layer(s) of the model to be inspected by the Caricature visualization
        input_key (str): A string key corresponding to the tensor to be passed to the model
        n_inputs (int): How many samples should be drawn from the input_key tensor for visualization
        decode_dictionary (dict): A dictionary mapping model outputs to class names
        output_name (str): The key which the caricature image will be saved into within the state dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        n_steps (int): How many steps of optimization to run when computing caricatures (quality vs time trade)
        learning_rate (float): The learning rate of the caricature optimizer. Should be higher than usual
        blur (float): How much blur to add to images during caricature generation
        cossim_pow (float): How much should similarity in form be valued versus creative license
        sd (float): The standard deviation of the noise used to seed the caricature
        fft (bool): Whether to use fft space (True) or image space (False) to create caricatures
        decorrelate (bool): Whether to use an ImageNet-derived color correlation matrix to de-correlate colors in \
                            the caricature. Parameter has no effect on grey scale images.
        sigmoid (bool): Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values
    """
    def __init__(self,
                 model_name,
                 layer_ids,
                 input_key,
                 n_inputs=1,
                 decode_dictionary=None,
                 output_name=None,
                 im_freq=1,
                 n_steps=128,
                 learning_rate=0.05,
                 blur=1,
                 cossim_pow=0.5,
                 sd=0.01,
                 fft=True,
                 decorrelate=True,
                 sigmoid=True):

        self.data = []
        self.in_key = input_key
        if output_name is None:
            output_name = "{}_caricature".format(model_name)
        super().__init__(inputs=self.in_key, outputs=output_name, mode='eval')
        self.n_inputs = n_inputs
        self.output_key = output_name
        self.im_freq = im_freq
        self.recording = False
        self.model_name = model_name
        self.model = None
        self.layer_ids = to_list(layer_ids)
        self.decode_dictionary = decode_dictionary
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.blur = blur
        self.cossim_pow = cossim_pow,
        self.sd = sd
        self.fft = fft
        self.decorrelate = decorrelate
        self.sigmoid = sigmoid

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

        with Suppressor():
            fig = plot_caricature(self.model,
                                  self.data,
                                  self.layer_ids,
                                  decode_dictionary=self.decode_dictionary,
                                  n_steps=self.n_steps,
                                  learning_rate=self.learning_rate,
                                  blur=self.blur,
                                  cossim_pow=self.cossim_pow,
                                  sd=self.sd,
                                  fft=self.fft,
                                  decorrelate=self.decorrelate,
                                  sigmoid=self.sigmoid)
        # TODO - Figure out how to get this to work without it displaying the figure. maybe fig.canvas.draw
        plt.draw()
        plt.pause(0.000001)
        state.maps[1][self.output_key] = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                                       sep='').reshape((1, ) + fig.canvas.get_width_height()[::-1] +
                                                                       (3, ))
        plt.close(fig)
