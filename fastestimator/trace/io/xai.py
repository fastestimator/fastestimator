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
import tensorflow as tf

from fastestimator.trace.trace import Trace


class XAiTrace(Trace):
    """
    Args:
        model_name (str): The model to be inspected by the visualization
        model_input (str, tf.Tensor): The input to the model, either a string key or the actual input tensor
        n_inputs (int): How many inputs to be collected and passed to the model (if model_input is a string)
        resample_inputs (bool): Whether to re-sample inputs every im_freq iterations or use the same throughout training
                                Can only be True if model_input is a string
        output_key (str): The name of the output to be written into the batch dictionary
        im_freq (int): Frequency (in epochs) during which visualizations should be generated
        mode (str): The mode ('train', 'eval') on which to run the trace
    """
    def __init__(self,
                 model_name,
                 model_input,
                 n_inputs=1,
                 resample_inputs=False,
                 output_key=None,
                 im_freq=1,
                 mode="eval"):
        if isinstance(model_input, str):  # Get inputs from key during training
            self.input_key = model_input
            self.collected_inputs = {"train": 0, "eval": 0}
            self.n_inputs = n_inputs
            self.data = {"train": [], "eval": []}
            self.resample_inputs = resample_inputs
        else:  # Inputs are provided as a tensor
            self.input_key = None
            self.n_inputs = model_input.shape[0]
            self.data = {mode or "train": model_input, mode or "eval": model_input}
            self.collected_inputs = {key: val.shape[0] for key, val in self.data.items()}
            self.resample_inputs = False
        if output_key is None:
            self.output_key = "{}_{}".format(model_name, type(self).__name__)
        super().__init__(inputs=self.input_key, outputs=output_key, mode=mode)
        self.model_name = model_name
        self.model = None
        self.im_freq = im_freq
        self.force_reset = False

    def on_begin(self, state):
        self.model = self.network.model[self.model_name]

    def on_epoch_begin(self, state):
        if self.force_reset or (self.resample_inputs and self._data_epoch(state['epoch'])):
            self.collected_inputs[state['mode']] = 0
            self.data[state['mode']] = []
            self.force_reset = False

    def on_batch_end(self, state):
        if self._data_epoch(state['epoch']) and self.collected_inputs[state['mode']] < self.n_inputs:
            samples = state.get(self.input_key) or state["batch"][self.input_key]
            self.collected_inputs[state['mode']] += samples.shape[0]
            self.data[state['mode']].append(samples)

    def on_epoch_end(self, state):
        if self._data_epoch(state['epoch']):
            self.data[state['mode']] = tf.concat(self.data[state['mode']], axis=0)
            self.data[state['mode']] = self.data[state['mode']][:self.n_inputs]
            if self.collected_inputs[state['mode']] < self.n_inputs:  # There aren't enough inputs to satisfy n_inputs
                self.force_reset = True

    def _data_epoch(self, epoch):
        return self.input_key and (epoch == 0 or (self.resample_inputs and epoch % self.im_freq == 0))
