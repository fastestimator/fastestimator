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
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.markers import MarkerStyle
from scipy.ndimage.filters import gaussian_filter1d

from fastestimator.summary.summary import Summary
from fastestimator.util.util import prettify_metric_name, to_list, to_set


class _MetricGroup:
    """A class for wrapping the values recorded for a given metric based on its experiment id and mode.

    This class is intentionally not @traceable.
    """
    state: DefaultDict[int, Dict[str, np.ndarray]]

    def __init__(self):
        self.state = defaultdict(dict)  # exp_id: {mode: value}

    def __getitem__(self, exp_id: int) -> Dict[str, np.ndarray]:
        return self.state[exp_id]

    def add(self, exp_id: int, mode: str, values: Dict[int, Any]) -> bool:
        """Add a new set of values to the metric group.

        Args:
            exp_id: The experiment id associated with these `values`.
            mode: The mode associated with these `values`.
            values: A dictionary of time: value pairs

        Returns:
            Whether the add was successful.
        """
        if values:
            values = list(sorted(values.items()))
            if len(values) == 1:
                # We will allow any data types if there's only one value since it will be displayed differently
                self.state[exp_id][mode] = np.array(values)
                return True
            else:
                # We will be plotting something over time
                for idx, (step, elem) in enumerate(values):
                    if isinstance(elem, np.ndarray):
                        if elem.ndim == 0 or (elem.ndim == 1 and elem.shape[0] == 1):
                            elem = elem.item()
                        else:
                            # TODO - handle larger arrays somehow (maybe a heat map?)
                            return False
                    if isinstance(elem, str):
                        # Can't plot strings over time...
                        elem = [float(s) for s in re.findall(r'(\d+\.\d+|\.?\d+)', elem)]
                        if len(elem) == 1:
                            # We got an unambiguous number
                            elem = elem[0]
                        else:
                            # Can't disambiguate what should be plotted
                            return False
                    if not isinstance(elem, (int, float)):
                        # Can only plot numeric values over time
                        return False
                    values[idx] = (step, elem)
                self.state[exp_id][mode] = np.array(values)

    def ndim(self) -> int:
        """Compute how many dimensions this data require to plot.

        Returns:
            The number of dimensions this data requires to plot.
        """
        ndims = [0]
        for mode_val in self.state.values():
            for mode, values in mode_val.items():
                if values.ndim in (0, 1):
                    # A singular value (this should never happen based on implementation of summary)
                    ndims.append(1)
                elif values.ndim == 2:
                    if values.shape[0] == 1:
                        # Metrics with only 1 time point can be displayed as singular values
                        ndims.append(1)
                    else:
                        # A regular time vs metric value plot
                        ndims.append(2)
                else:
                    # Time vs array value. Not supported yet.
                    ndims.append(3)
        return max(ndims)

    def get_val(self, exp_idx: int, mode: str) -> Union[None, str, np.ndarray]:
        """Get the value for a given experiment id and mode.

        Args:
            exp_idx: The id of the experiment in question.
            mode: The mode under consideration.

        Returns:
            The value associated with the `exp_id` and `mode`, or None if no such value exists. If only a single item
            exists and it is numeric then it will be returned as a string truncated to 5 decimal places.
        """
        vals = self.state[exp_idx].get(mode, None)
        if vals is None:
            return vals
        if vals.ndim in (0, 1):
            item = vals.item()
            if isinstance(item, float):
                return "{:.5f}".format(item)
            return str(item)
        if vals.ndim == 2 and vals.shape[0] == 1:
            # This value isn't really time dependent
            item = vals[0][1]
            if isinstance(item, float):
                return "{:.5f}".format(item)
            return str(item)
        else:
            return vals

    def modes(self, exp_id: int) -> List[str]:
        """Get the modes supported by this group for a given experiment.

        Args:
            exp_id: The id of the experiment in question.

        Returns:
            Which modes have data for the given `exp_id`.
        """
        return list(self.state[exp_id].keys())


def plot_logs(experiments: List[Summary],
              smooth_factor: float = 0,
              share_legend: bool = True,
              ignore_metrics: Optional[Set[str]] = None,
              pretty_names: bool = False,
              include_metrics: Optional[Set[str]] = None) -> plt.Figure:
    """A function which will plot experiment histories for comparison viewing / analysis.

    Args:
        experiments: Experiment(s) to plot.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        ignore_metrics: Any keys to ignore during plotting.
        include_metrics: A whitelist of keys to include during plotting. If None then all will be included.

    Returns:
        The handle of the pyplot figure.
    """
    experiments = to_list(experiments)
    n_experiments = len(experiments)
    if n_experiments == 0:
        return plt.subplots(111)[0]

    ignore_keys = ignore_metrics or set()
    ignore_keys = to_set(ignore_keys)
    ignore_keys |= {'epoch'}
    include_keys = to_set(include_metrics)
    # TODO: epoch should be indicated on the axis (top x axis?). Problem - different epochs per experiment.
    # TODO: figure out how ignore_metrics should interact with mode

    metric_histories = defaultdict(_MetricGroup)  # metric: MetricGroup
    for idx, experiment in enumerate(experiments):
        history = experiment.history
        # Since python dicts remember insertion order, sort the history so that train mode is always plotted on bottom
        for mode, metrics in sorted(history.items(),
                                    key=lambda x: 0 if x[0] == 'train' else 1 if x[0] == 'eval' else 2 if x[0] == 'test'
                                    else 3 if x[0] == 'infer' else 4):
            for metric, step_val in metrics.items():
                if len(step_val) == 0:
                    continue  # Ignore empty metrics
                if metric in ignore_keys:
                    continue
                if include_keys and metric not in include_keys:
                    continue
                metric_histories[metric].add(idx, mode, step_val)

    metric_list = list(sorted(metric_histories.keys()))
    if len(metric_list) == 0:
        return plt.subplots(111)[0]

    # If sharing legend and there is more than 1 plot, then dedicate 1 subplot for the legend
    share_legend = share_legend and (len(metric_list) > 1)
    n_legends = math.ceil(n_experiments / 4)
    n_plots = len(metric_list) + (share_legend * n_legends)

    # map the metrics into an n x n grid, then remove any extra columns. Final grid will be n x m with m <= n
    n_rows = math.ceil(math.sqrt(n_plots))
    n_cols = math.ceil(n_plots / n_rows)
    metric_grid_location = {}
    nd1_metrics = []
    idx = 0
    for metric in metric_list:
        if metric_histories[metric].ndim() == 1:
            # Delay placement of the 1D plots until the end
            nd1_metrics.append(metric)
        else:
            metric_grid_location[metric] = (idx // n_cols, idx % n_cols)
            idx += 1
    for metric in nd1_metrics:
        metric_grid_location[metric] = (idx // n_cols, idx % n_cols)
        idx += 1

    sns.set_context('paper')
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', figsize=(4 * n_cols, 2.8 * n_rows))

    # If only one row, need to re-format the axs object for consistency. Likewise for columns
    if n_rows == 1:
        axs = [axs]
        if n_cols == 1:
            axs = [axs]

    for metric in metric_grid_location.keys():
        axis = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]]
        if metric_histories[metric].ndim() == 1:
            axis.grid(linestyle='')
        else:
            axis.grid(linestyle='--')
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        axis.set_title(metric if not pretty_names else prettify_metric_name(metric), fontweight='bold')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.tick_params(bottom=False, left=False)

    # some of the later rows/columns might be unused or reserved for legends, so disable them
    last_row_idx = math.ceil(len(metric_list) / n_cols) - 1
    last_column_idx = len(metric_list) - last_row_idx * n_cols - 1
    for c in range(n_cols):
        if c <= last_column_idx:
            axs[last_row_idx][c].set_xlabel('Steps')
            axs[last_row_idx][c].xaxis.set_tick_params(which='both', labelbottom=True)
        else:
            axs[last_row_idx][c].axis('off')
            axs[last_row_idx - 1][c].set_xlabel('Steps')
            axs[last_row_idx - 1][c].xaxis.set_tick_params(which='both', labelbottom=True)
        for r in range(last_row_idx + 1, n_rows):
            axs[r][c].axis('off')

    # the 1D metrics don't need x axis, so move them up, starting with the last in case multiple rows of them
    for metric in reversed(nd1_metrics):
        row = metric_grid_location[metric][0]
        col = metric_grid_location[metric][1]
        axs[row][col].axis('off')
        if row > 0:
            axs[row - 1][col].set_xlabel('Steps')
            axs[row - 1][col].xaxis.set_tick_params(which='both', labelbottom=True)

    colors = sns.hls_palette(n_colors=n_experiments, s=0.95) if n_experiments > 10 else sns.color_palette("colorblind")
    color_offset = defaultdict(lambda: 0)
    # If there is only 1 experiment, we will use alternate colors based on mode
    if n_experiments == 1:
        color_offset['eval'] = 1
        color_offset['test'] = 2
        color_offset['infer'] = 3

    handles = []
    labels = []
    has_label = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: False)))  # exp_id : {mode: {type: True}}
    ax_text = defaultdict(lambda: (0.0, 0.9))  # Where to put the text on a given axis
    for exp_idx, experiment in enumerate(experiments):
        for metric, group in metric_histories.items():
            axis = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]]
            if group.ndim() == 1:
                # Single value
                for mode in group.modes(exp_idx):
                    ax_id = id(axis)
                    prefix = f"{experiment.name} ({mode})" if n_experiments > 1 else f"{mode}"
                    axis.text(ax_text[ax_id][0],
                              ax_text[ax_id][1],
                              f"{prefix}: {group.get_val(exp_idx, mode)}",
                              color=colors[exp_idx + color_offset[mode]],
                              transform=axis.transAxes)
                    ax_text[ax_id] = (ax_text[ax_id][0], ax_text[ax_id][1] - 0.1)
                    if ax_text[ax_id][1] < 0:
                        ax_text[ax_id] = (ax_text[ax_id][0] + 0.5, 0.9)
            elif group.ndim() == 2:
                for mode, data in group[exp_idx].items():
                    title = f"{experiment.name} ({mode})" if n_experiments > 1 else f"{mode}"
                    if data.shape[0] < 2:
                        # This particular mode only has a single data point, so need to draw a shape instead of a line
                        xy = (data[0][0], data[0][1])
                        if mode == 'train':
                            style = MarkerStyle(marker='o', fillstyle='full')
                        elif mode == 'eval':
                            style = MarkerStyle(marker='v', fillstyle='full')
                        elif mode == 'test':
                            style = MarkerStyle(marker='*', fillstyle='full')
                        else:
                            style = MarkerStyle(marker='s', fillstyle='full')
                        s = axis.scatter(xy[0],
                                         xy[1],
                                         s=40,
                                         c=[colors[exp_idx + color_offset[mode]]],
                                         marker=style,
                                         linewidth=1.0,
                                         edgecolors='black',
                                         zorder=3)  # zorder to put markers on top of line segments
                        if not has_label[exp_idx][mode]['patch']:
                            labels.append(title)
                            handles.append(s)
                            has_label[exp_idx][mode]['patch'] = True
                    else:
                        # We can draw a line
                        y = data[:, 1] if smooth_factor == 0 else gaussian_filter1d(data[:, 1], sigma=smooth_factor)
                        ln = axis.plot(
                            data[:, 0],
                            y,
                            color=colors[exp_idx + color_offset[mode]],
                            label=title,
                            linewidth=1.5,
                            linestyle='solid' if mode == 'train' else
                            'dashed' if mode == 'eval' else 'dotted' if mode == 'test' else 'dashdot')
                        if not has_label[exp_idx][mode]['line']:
                            labels.append(title)
                            handles.append(ln[0])
                            has_label[exp_idx][mode]['line'] = True
            else:
                # Some kind of image or matrix. Not implemented yet.
                pass

    plt.tight_layout()

    if labels:
        if share_legend:
            # Sort the labels
            handles = [h for _, h in sorted(zip(labels, handles), key=lambda pair: pair[0])]
            labels = sorted(labels)
            # Split the labels over multiple legends if there are too many to fit in one axis
            elems_per_legend = math.ceil(len(labels) / n_legends)
            i = 0
            for r in range(last_row_idx, n_rows):
                for c in range(last_column_idx + 1 if r == last_row_idx else 0, n_cols):
                    axs[r][c].legend(
                        handles[i:i + elems_per_legend],
                        labels[i:i + elems_per_legend],
                        loc='center',
                        fontsize='large' if elems_per_legend <= 6 else 'medium' if elems_per_legend <= 8 else 'small')
                    i += elems_per_legend
        else:
            for i in range(n_rows):
                for j in range(n_cols):
                    if i == last_row_idx and j > last_column_idx:
                        break
                    axs[i][j].legend(loc='best', fontsize='small')
    return fig


def visualize_logs(experiments: List[Summary],
                   save_path: str = None,
                   smooth_factor: float = 0,
                   share_legend: bool = True,
                   pretty_names: bool = False,
                   ignore_metrics: Optional[Set[str]] = None,
                   include_metrics: Optional[Set[str]] = None,
                   verbose: bool = True):
    """A function which will save or display experiment histories for comparison viewing / analysis.

    Args:
        experiments: Experiment(s) to plot.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        ignore_metrics: Any metrics to ignore during plotting.
        include_metrics: A whitelist of metric keys (None whitelists all keys).
        verbose: Whether to print out the save location.
    """
    plot_logs(experiments,
              smooth_factor=smooth_factor,
              share_legend=share_legend,
              pretty_names=pretty_names,
              ignore_metrics=ignore_metrics,
              include_metrics=include_metrics)
    if save_path is None:
        plt.show()
    else:
        save_path = os.path.normpath(save_path)
        root_dir = os.path.dirname(save_path)
        if root_dir == "":
            root_dir = "."
        os.makedirs(root_dir, exist_ok=True)
        save_file = os.path.join(root_dir, os.path.basename(save_path) or 'parse_logs.png')
        if verbose:
            print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
