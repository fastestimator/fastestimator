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
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

from fastestimator.summary.summary import Summary
from fastestimator.util.util import prettify_metric_name, to_list, to_set


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

    ignore_keys = ignore_metrics or set()
    ignore_keys = to_set(ignore_keys)
    ignore_keys |= {'epoch', 'progress', 'total_train_steps'}
    include_keys = to_set(include_metrics)
    # TODO: epoch should be indicated on the axis (top x axis?)
    # TODO: figure out how ignore_metrics should interact with mode

    max_time = 0
    metric_keys = set()
    for experiment in experiments:
        history = experiment.history
        for mode, metrics in history.items():
            for key, value in metrics.items():
                if key in ignore_keys:
                    continue
                if include_keys and key not in include_keys:
                    ignore_keys.add(key)
                    continue
                if any(map(lambda x: isinstance(x[1], np.ndarray) and x[1].ndim > 0, value.items())):
                    ignore_keys.add(key)
                    continue  # TODO: nd array not currently supported. maybe in future visualize as heat map?
                if value.keys():
                    max_time = max(max_time, max(value.keys()))
                metric_keys.add("{}: {}".format(mode, key))
    metric_list = sorted(list(metric_keys))  # Sort the metrics alphabetically for consistency
    num_metrics = len(metric_list)
    num_experiments = len(experiments)

    if num_metrics == 0:
        return plt.subplots(111)[0]

    # map the metrics into an n x n grid, then remove any extra rows. Final grid will be m x n with m <= n
    num_cols = math.ceil(math.sqrt(num_metrics))
    metric_grid_location = {key: (idx // num_cols, idx % num_cols) for (idx, key) in enumerate(metric_list)}
    num_rows = math.ceil(num_metrics / num_cols)

    sns.set_context('paper')
    fig, axs = plt.subplots(num_rows, num_cols, sharex='all', figsize=(4 * num_cols, 2.8 * num_rows))

    # If only one row, need to re-format the axs object for consistency. Likewise for columns
    if num_rows == 1:
        axs = [axs]
        if num_cols == 1:
            axs = [axs]

    for metric in metric_grid_location.keys():
        axis = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]]
        axis.set_title(metric if not pretty_names else prettify_metric_name(metric))
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        axis.grid(linestyle='--')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.tick_params(bottom=False, left=False)

    for i in range(num_cols):
        axs[num_rows - 1][i].set_xlabel('Steps')

    # some of the columns in the last row might be unused, so disable them
    last_column_idx = num_cols - (num_rows * num_cols - num_metrics) - 1
    for i in range(last_column_idx + 1, num_cols):
        axs[num_rows - 1][i].axis('off')
        axs[num_rows - 2][i].set_xlabel('Steps')
        axs[num_rows - 2][i].xaxis.set_tick_params(which='both', labelbottom=True)

    colors = sns.hls_palette(n_colors=num_experiments,
                             s=0.95) if num_experiments > 10 else sns.color_palette("colorblind")

    handles = []
    labels = []
    bar_counter = defaultdict(lambda: 0)
    for (color_idx, experiment) in enumerate(experiments):
        labels.append(experiment.name)
        metrics = {
            "{}: {}".format(mode, key): val
            for mode,
            sub in experiment.history.items() for key,
            val in sub.items() if key not in ignore_keys
        }
        for (idx, (metric, value)) in enumerate(metrics.items()):
            data = np.array(list(value.items()))
            if len(data) == 1:
                y = data[0][1]
                if isinstance(y, str):
                    vals = [float(x) for x in re.findall(r'\d+\.?\d+', y)]
                    if len(vals) == 1:
                        y = vals[0]
                width = max(10, max_time // 10)
                x = max_time // 2 + (2 * (bar_counter[metric] % 2) - 1) * width * math.ceil(bar_counter[metric] / 2)
                ln = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]].bar(
                    x=x, height=y, color=colors[color_idx], label=experiment.name, width=width)
                bar_counter[metric] += 1
            else:
                y = data[:, 1] if smooth_factor == 0 else gaussian_filter1d(data[:, 1], sigma=smooth_factor)
                ln = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]].plot(
                    data[:, 0], y, color=colors[color_idx], label=experiment.name, linewidth=1.5)
            if idx == 0:
                handles.append(ln[0])

    plt.tight_layout()

    if len(labels) > 1 or labels[0]:
        if share_legend and num_rows > 1:
            if last_column_idx == num_cols - 1:
                fig.subplots_adjust(bottom=0.15)
                fig.legend(handles, labels, loc='lower center', ncol=num_cols + 1)
            else:
                axs[num_rows - 1][last_column_idx + 1].legend(handles, labels, loc='center', fontsize='large')
        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    if i == num_rows - 1 and j > last_column_idx:
                        break
                    axs[i][j].legend(loc='best', fontsize='small')
    return fig


def visualize_logs(experiments: List[Summary],
                   save_path: str = None,
                   smooth_factor: float = 0,
                   share_legend: bool = True,
                   pretty_names: bool = False,
                   ignore_metrics: Optional[Set[str]] = None,
                   include_metrics: Optional[Set[str]] = None):
    """A function which will save or display experiment histories for comparison viewing / analysis.

    Args:
        experiments: Experiment(s) to plot.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        ignore_metrics: Any metrics to ignore during plotting.
        include_metrics: A whitelist of metric keys (None whitelists all keys).
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
        save_path = os.path.dirname(save_path)
        if save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'parse_logs.png')
        print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
