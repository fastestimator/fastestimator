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

from fastestimator.trace.io.xai import XAiTrace
from fastestimator.util.util import to_list, Suppressor
from fastestimator.xai import plot_caricature, fig_to_img


class Caricature(XAiTrace):
    """
    Args:
        model_name (str): The model to be inspected by the Caricature visualization
        layer_ids (int, list): The layer(s) of the model to be inspected by the Caricature visualization
        model_input (str, tf.Tensor): The input to the model, either a string key or the actual input tensor
        n_inputs (int): How many samples should be drawn from the input_key tensor for visualization
        resample_inputs (bool): Whether to re-sample inputs every im_freq iterations or use the same throughout training
                                Can only be True if model_input is a string
        output_key (str): The name of the output to be written into the batch dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        mode (str): The mode ('train', 'eval') on which to run the trace
        decode_dictionary (dict): A dictionary mapping model outputs to class names
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
                 model_input,
                 n_inputs=1,
                 resample_inputs=False,
                 output_key=None,
                 im_freq=1,
                 mode="eval",
                 decode_dictionary=None,
                 n_steps=128,
                 learning_rate=0.05,
                 blur=1,
                 cossim_pow=0.5,
                 sd=0.01,
                 fft=True,
                 decorrelate=True,
                 sigmoid=True):
        super().__init__(model_name=model_name,
                         model_input=model_input,
                         n_inputs=n_inputs,
                         resample_inputs=resample_inputs,
                         output_key=output_key,
                         im_freq=im_freq,
                         mode=mode)

        self.layer_ids = to_list(layer_ids)
        self.decode_dictionary = decode_dictionary
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.blur = blur
        self.cossim_pow = cossim_pow
        self.sd = sd
        self.fft = fft
        self.decorrelate = decorrelate
        self.sigmoid = sigmoid

    def on_epoch_end(self, state):
        super().on_epoch_end(state)
        if state['epoch'] % self.im_freq != 0:
            return
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        with Suppressor():
            fig = plot_caricature(self.model,
                                  self.data[state['mode']],
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
        fig.canvas.draw()
        state.maps[1][self.output_key] = fig_to_img(fig)
        matplotlib.use(old_backend)
