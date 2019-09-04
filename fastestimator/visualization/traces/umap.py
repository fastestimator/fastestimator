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
import seaborn as sns
import tensorflow as tf
import umap
from matplotlib.lines import Line2D

from fastestimator.estimator.trace import Trace
from fastestimator.util.util import Suppressor


class UMap(Trace):
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
        self.fit = umap.UMAP(**umap_parameters)
        self.n_components = umap_parameters.get("n_components", 2)
        self.data = []
        self.labels = []
        self.color_dict = None
        self.label_dict = label_dict
        self.legend_elems = None
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
        color_list = self._map_classes_to_colors(tf.concat(self.labels, axis=0))
        if self.legend_elems is None and color_list is not None:
            self.legend_elems = [
                Line2D([0], [0],
                       marker='o',
                       color='w',
                       markerfacecolor=self.color_dict[clazz],
                       label=clazz if self.label_dict is None else self.label_dict[clazz],
                       markersize=7) for clazz in self.color_dict
            ]
        with Suppressor():  # Silence a bunch of numba warnings
            points = self.fit.fit_transform(tf.concat(self.data, axis=0))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks([], [])
        ax.set_yticklabels([])
        ax.set_xticks([], [])
        ax.set_xticklabels([])
        if self.n_components == 1:
            ax.scatter(points[:, 0], range(len(points)), c=color_list or 'b', s=3)
        if self.n_components == 2:
            ax.scatter(points[:, 0], points[:, 1], c=color_list or 'b', s=3)
        if self.n_components == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_list or 'b', s=3)
        if self.legend_elems and self.legend_loc != 'off':
            ax.legend(handles=self.legend_elems, loc=self.legend_loc, fontsize='small')
        plt.tight_layout()
        # TODO - Figure out how to get this to work without it displaying the figure. maybe fig.canvas.draw
        plt.draw()
        plt.pause(0.000001)
        state.maps[1][self.output_key] = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                                       sep='').reshape((1, ) + fig.canvas.get_width_height()[::-1] +
                                                                       (3, ))
        plt.close(fig)
        self.data.clear()
        self.labels.clear()

    def _map_classes_to_colors(self, classifications):
        if classifications is None or len(classifications) == 0:
            return None
        if self.color_dict is None:
            classes = set(map(lambda x: int(x), classifications))
            num_classes = len(classes)
            colors = sns.hls_palette(n_colors=num_classes,
                                     s=0.95) if num_classes > 10 else sns.color_palette("colorblind")
            class_to_color = {clazz: idx for idx, clazz in enumerate(classes)}
            self.color_dict = {clazz: colors[class_to_color[clazz]] for clazz in classes}
        return [self.color_dict[int(clazz)] for clazz in classifications]
