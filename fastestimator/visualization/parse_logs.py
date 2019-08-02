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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

from fastestimator.util.util import prettify_metric_name, remove_blacklist_keys, strip_suffix
from fastestimator.util.loader import PathLoader


def parse_file(file_path):
    """
    A function which will parse log files into a dictionary of metrics

    Args:
        file_path: The path to a log file

    Returns:
        A dictionary of the form {"metric_name":[[<step>,<value>],...]}
    """
    metrics = defaultdict(list)
    with open(file_path) as file:
        for line in file:
            if not line.startswith("FastEstimator-"):
                continue
            parsed_line = re.findall(r"([^:^;\s]+):[\s]*([-]?[0-9]+[.]?[0-9]*);", line)
            step = parsed_line[0]
            assert step[0] == "step", \
                "Log file (%s) seems to be missing step information, or step is not listed first" % file
            for metric in parsed_line[1:]:
                metrics[metric[0]].append([int(step[1]), float(metric[1])])
    return metrics


def graph_metrics(files_metrics, smooth_factor, save, save_path, share_legend, pretty_names):
    """
    A function which will plot (and optionally save) file metrics for comparison viewing / analysis

    Args:
        files_metrics: A dictionary of the form {"file_name":{"metric_name":[[<step>,<value>],...]}}
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None

    Side Effects:
        Generates and either displays or saves a graph file
    """
    assert smooth_factor >= 0, "smooth factor must be non-negative (zero to disable)"
    assert len(files_metrics) > 0, "must have at least one file to plot"

    metric_keys = {keys for files in files_metrics.values() for keys in files.keys()}
    metric_list = list(metric_keys)
    metric_list.sort()  # Sort the metrics alphabetically for consistency
    num_metrics = len(metric_list)
    num_experiments = len(files_metrics)

    assert num_metrics > 0, "must have at least one metric to plot"

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

    colors = sns.hls_palette(n_colors=num_experiments, s=0.95) if num_experiments > 10 else sns.color_palette(
        "colorblind")

    handles = []
    labels = []
    for (color_idx, file) in enumerate(files_metrics):
        labels.append(file)
        for (idx, metric) in enumerate(files_metrics[file]):
            data = np.array(files_metrics[file][metric])
            y = data[:, 1] if smooth_factor == 0 else gaussian_filter1d(data[:, 1], sigma=smooth_factor)
            ln = axs[metric_grid_location[metric][0]][metric_grid_location[metric][1]].plot(data[:, 0], y,
                                                                                            color=colors[color_idx],
                                                                                            label=file,
                                                                                            linewidth=1.5)
            if idx == 0:
                handles.append(ln[0])

    plt.tight_layout()

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

    if not save:
        plt.show()
    else:
        save_path = os.path.dirname(save_path)
        if save_path is None or save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'parse_logs.png')
        print("Saving to %s" % save_file)
        plt.savefig(save_file, dpi=300)


def parse_files(file_paths, log_extension='.txt', smooth_factor=0, save=False, save_path=None, ignore_metrics=None,
                share_legend=True, pretty_names=False):
    """
    A function which will iterate through the given log file paths, parse them to extract metrics, remove any
        metrics which are blacklisted, and then pass the necessary information on the graphing function

    Args:
        file_paths: A list of paths to various log files
        log_extension: The extension of the log files
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided
        ignore_metrics: Any metrics within the log files which will not be visualized
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None
    """
    if file_paths is None or len(file_paths) < 1:
        raise AssertionError("must provide at least one log file")

    files_metrics = {}
    for file_path in file_paths:
        file_metrics = parse_file(file_path)
        remove_blacklist_keys(file_metrics, ignore_metrics)
        files_metrics[strip_suffix(os.path.split(file_path)[1].strip(), log_extension)] = file_metrics
    graph_metrics(files_metrics, smooth_factor, save,
                  save_path if save_path is not None else file_paths[0], share_legend, pretty_names)


def parse_folder(dir_path, log_extension='.txt', recursive_search=False, smooth_factor=1, save=False, save_path=None,
                 ignore_metrics=None, share_legend=True, pretty_names=False):
    """
    A function which will gather all log files within a given folder and pass them along for visualization

    Args:
        dir_path: The path to a directory containing log files
        log_extension: The extension of the log files
        recursive_search: Whether to recursively search sub-directories for log files
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply(zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided
        ignore_metrics: Any metrics within the log files which will not be visualized
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None
    """
    loader = PathLoader(dir_path, input_extension=log_extension, recursive_search=recursive_search)
    file_paths = [x[0] for x in loader.path_pairs]

    parse_files(file_paths, log_extension, smooth_factor, save, save_path, ignore_metrics, share_legend, pretty_names)
