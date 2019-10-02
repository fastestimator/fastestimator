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

from fastestimator.interpretation import UmapPlotter
from fastestimator.trace import Trace


class UMap(Trace):
    # TODO: let this take a model+layer instead of restricting it to just an existing key
    def __init__(self,
                 in_vector_key,
                 label_vector_key=None,
                 label_dict=None,
                 output_name=None,
                 legend_loc='best',
                 im_freq=1,
                 **umap_parameters):
        """
        Args:
            in_vector_key: The key of the input to be fed into the umap algorithm
            label_vector_key: The (optional) key of the classes corresponding to the inputs (used for coloring points)
            label_dict: An (optional) dictionary mapping labels from the label vector to other representations
                        (ex. {0:'dog', 1:'cat'})
            output_name: The key which the umap image will be saved into within the state dictionary
            legend_loc: The location of the legend, or 'off' to disable figure legends
            im_freq: Frequency (in epochs) during which visualizations should be generated
            **umap_parameters: Extra parameters to be passed to the umap algorithm, ex. n_neighbors, n_epochs, etc.
        """
        if output_name is None:
            output_name = "{}_umap".format(in_vector_key)
        super().__init__(inputs={in_vector_key, label_vector_key}, outputs=output_name, mode='eval')
        self.in_key = in_vector_key
        self.label_key = label_vector_key
        self.output_key = output_name
        self.umap = UmapPlotter(label_dict=label_dict, **umap_parameters)
        self.data = []
        self.labels = []
        self.legend_loc = legend_loc
        self.im_freq = im_freq
        self.recording = False

    def on_epoch_begin(self, state):
        if state['epoch'] % self.im_freq == 0:
            self.recording = True
        else:
            self.recording = False

    def on_batch_end(self, state):
        if self.recording:
            self.data.append(state.get(self.in_key) or state['batch'][self.in_key])
            if self.label_key:
                self.labels.append(state.get(self.label_key) or state['batch'][self.label_key])

    def on_epoch_end(self, state):
        if not self.recording:
            return
        fig = self.umap.plot_umap(tf.concat(self.data, axis=0),
                                  tf.concat(self.labels, axis=0),
                                  legend_loc=self.legend_loc,
                                  title=self.in_key)
        # TODO - Figure out how to get this to work without it displaying the figure. maybe fig.canvas.draw
        plt.draw()
        plt.pause(0.000001)
        state.maps[1][self.output_key] = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                                       sep='').reshape((1, ) + fig.canvas.get_width_height()[::-1] +
                                                                       (3, ))
        plt.close(fig)
        self.data.clear()
        self.labels.clear()
