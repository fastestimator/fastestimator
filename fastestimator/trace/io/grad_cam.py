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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from fastestimator.interpretation import plot_gradcam
from fastestimator.trace import Trace


class GradCam(Trace):
    def __init__(self,
                 model_name,
                 input_key,
                 n_inputs=1,
                 layer_id=None,
                 output_name=None,
                 im_freq=1,
                 decode_dictionary=None,
                 color_map=cv2.COLORMAP_INFERNO):

        self.input_data = []
        self.label_data = []
        self.in_key = input_key
        if output_name is None:
            output_name = "{}_gradCam".format(model_name)
        super().__init__(inputs=self.in_key, outputs=output_name, mode='eval')
        self.n_inputs = n_inputs
        self.output_key = output_name
        self.model_name = model_name
        self.model = None
        self.im_freq = im_freq
        self.color_map = color_map
        self.layer_id = layer_id
        self.decode_dictionary = decode_dictionary

    def on_begin(self, state):
        self.model = self.network.model[self.model_name]
        if isinstance(self.layer_id, int):
            self.layer_id = self.model.layers[self.layer_id].name
        if self.layer_id is None:
            for layer in reversed(self.model.layers):
                if layer.output.shape.ndims == 4:
                    self.layer_id = layer.name
                    break
        assert self.model.get_layer(self.layer_id).output.shape.ndims == 4, \
            "GradCam must run on a layer with outputs of the form [batch, height, width, filters] (a convolution layer)"

    def on_batch_end(self, state):
        if state['epoch'] == 0 and len(self.input_data) <= self.n_inputs:
            self.input_data.append(state.get(self.in_key) or state['batch'][self.in_key])

    def on_epoch_end(self, state):
        if state['epoch'] % self.im_freq != 0:
            return
        if state['epoch'] == 0:
            self.input_data = tf.concat(self.input_data, axis=0)
            self.input_data = self.input_data[:self.n_inputs]

        fig = plot_gradcam(self.input_data,
                           self.model,
                           self.layer_id,
                           decode_dictionary=self.decode_dictionary,
                           colormap=self.color_map)
        # TODO - Figure out how to get this to work without it displaying the figure. maybe fig.canvas.draw
        plt.draw()
        plt.pause(0.000001)
        flat_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        flat_image_pixels = flat_image.shape[0] // 3
        width, height = fig.canvas.get_width_height()
        if flat_image_pixels % height != 0:
            # Canvas returned incorrect width/height. This seems to happen sometimes in Jupyter. TODO: figure out why.
            search = 1
            guess = height + search
            while flat_image_pixels % guess != 0:
                if search < 0:
                    search = -1 * search + 1
                else:
                    search = -1 * search
                guess = height + search
            height = guess
            width = flat_image_pixels // height
        state.maps[1][self.output_key] = flat_image.reshape((1, height, width, 3))
        plt.close(fig)
