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
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
# noinspection PyPackageRequirements
import umap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from fastestimator.util.util import Suppressor

assert Axes3D  # Axes3D is used to enable projection='3d', but will show up as unused without the assert


# TODO: make cache useful again, and fix the progress messages
class UmapPlotter:
    def __init__(self, label_dict=None, **umap_parameters):
        self.fit = umap.UMAP(**umap_parameters)
        self.n_components = umap_parameters.get("n_components", 2)
        self.label_dict = label_dict
        self.legend_elems = None
        self.color_dict = None

    def _map_classes_to_colors(self, classifications):
        if classifications is None or len(classifications) == 0:
            return None
        if self.color_dict is None:
            classes = set(map(lambda x: x if not hasattr(x, 'numpy') else int(x), classifications))
            num_classes = len(classes)
            colors = sns.hls_palette(n_colors=num_classes,
                                     s=0.95) if num_classes > 10 else sns.color_palette("colorblind")
            class_to_color = {clazz: idx for idx, clazz in enumerate(classes)}
            self.color_dict = {clazz: colors[class_to_color[clazz]] for clazz in classes}
        return [self.color_dict[clazz if not hasattr(clazz, 'numpy') else int(clazz)] for clazz in classifications]

    def plot_umap(self, data, labels=None, legend_loc='best', title=None, fig_ax=None):
        color_list = self._map_classes_to_colors(labels)
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
            points = self.fit.fit_transform(data)

        if not fig_ax:
            fig = plt.figure(dpi=96)
            ax = fig.add_subplot(111)
        else:
            fig, ax = fig_ax
        ax.set_yticks([], [])
        ax.set_yticklabels([])
        ax.set_xticks([], [])
        ax.set_xticklabels([])
        if title:
            ax.set_title(title)
        if self.n_components == 1:
            ax.scatter(points[:, 0], range(len(points)), c=color_list or 'b', s=3)
        if self.n_components == 2:
            ax.scatter(points[:, 0], points[:, 1], c=color_list or 'b', s=3)
        if self.n_components == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_list or 'b', s=3)
        if self.legend_elems and legend_loc != 'off':
            ax.legend(handles=self.legend_elems, loc=legend_loc, fontsize='small')
        plt.tight_layout()
        return fig

    def visualize_umap(self, data, labels=None, legend_loc='best', save_path=None, title=None):
        if isinstance(data, list):
            if title is not None:
                assert isinstance(title, list) and len(title) == len(data), \
                    "If titles are provided, must be one-to-one with the data"
            num_layers = len(data)
            n_components = len(data[0][0])
            num_cols = math.ceil(math.sqrt(num_layers))
            num_rows = math.ceil(num_layers / num_cols)
            layer_grid_location = {idx: (idx // num_cols, idx % num_cols) for idx in range(num_layers)}
            if n_components == 3:
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.8 * num_rows),
                                        subplot_kw={'projection': '3d'})
            else:
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.8 * num_rows))
            # If only one row, need to re-format the axs object for consistency. Likewise for columns
            if num_rows == 1:
                axs = [axs]
                if num_cols == 1:
                    axs = [axs]
            # Turn off axis since numeric values are not meaningful
            for i in range(num_rows):
                for j in range(num_cols):
                    axis = axs[i][j]
                    axis.set_yticks([], [])
                    axis.set_yticklabels([])
                    axis.set_xticks([], [])
                    axis.set_xticklabels([])
            # some of the columns in the last row might be unused, so disable them
            last_column_idx = num_cols - (num_rows * num_cols - num_layers) - 1
            for i in range(last_column_idx + 1, num_cols):
                axs[num_rows - 1][i].axis('off')

            legend_loc_plot = legend_loc
            if legend_loc == 'shared':
                if num_rows == 1:
                    legend_loc_plot = 'best'
                else:
                    legend_loc_plot = 'off'
            for idx, elem in enumerate(tqdm(data, desc='Visualizing')):
                ax = axs[layer_grid_location[idx][0]][layer_grid_location[idx][1]]
                self.plot_umap(elem,
                               labels,
                               legend_loc=legend_loc_plot,
                               title=None if title is None else title[idx],
                               fig_ax=(fig, ax))
            if legend_loc == 'shared' and num_rows > 1:
                if last_column_idx == num_cols - 1:
                    fig.subplots_adjust(bottom=0.15)
                    fig.legend(handles=self.legend_elems, loc='lower center', ncol=num_cols + 1)
                else:
                    axs[num_rows - 1][last_column_idx + 1].legend(handles=self.legend_elems,
                                                                  loc='center',
                                                                  fontsize='large')
        else:
            self.plot_umap(data, labels, legend_loc=legend_loc, title=title)
        if save_path is None:
            plt.show()
        else:
            save_path = os.path.dirname(save_path)
            if save_path == "":
                save_path = "."
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'umap.png')
            print("Saving to {}".format(save_file))
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
