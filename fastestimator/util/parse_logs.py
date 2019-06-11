import argparse
import math
import os
import re
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


def strip_suffix(target, suffix):
    """
    Remove the given suffix from the target if it is present there

    Args:
        target: A string to be formatted
        suffix: A string to be removed from 'target'

    Returns:
        The formatted version of 'target'
    """
    if suffix is None or target is None:
        return target
    s_len = len(suffix)
    if target[-s_len:] == suffix:
        return target[:-s_len]
    return target


def prettify_metric_name(metric):
    """
    Add spaces to camel case words, then swap _ for space, and capitalize each word

    Args:
        metric: A string to be formatted

    Returns:
        The formatted version of 'metric'
    """
    return string.capwords(re.sub("([a-z])([A-Z])", r"\g<1> \g<2>", metric).replace("_", " "))


def remove_blacklist_keys(dic, blacklist):
    """
    A function which removes the blacklisted elements from a dictionary

    Args:
        dic: The dictionary to inspect
        blacklist: keys to be removed from dic if they are present

    Returns:
        None

    Side Effects:
        Entries in dic may be removed
    """
    if blacklist is None:
        return
    for elem in blacklist:
        dic.pop(elem, None)


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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = os.path.join(os.path.dirname(save_path), 'parse_logs.png')
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
    assert len(file_paths) > 0, "must provide at least one log file"

    files_metrics = {}
    for file_path in file_paths:
        file_metrics = parse_file(file_path)
        remove_blacklist_keys(file_metrics, ignore_metrics)
        files_metrics[strip_suffix(os.path.split(file_path)[1].strip(), log_extension)] = file_metrics
    graph_metrics(files_metrics, smooth_factor, save,
                  save_path if save_path is not None else file_paths[0], share_legend, pretty_names)


def parse_folder(dir_path, log_extension='.txt', smooth_factor=1, save=False, save_path=None, ignore_metrics=None,
                 share_legend=True, pretty_names=False):
    """
    A function which will gather all log files within a given folder and pass them along for visualization

    Args:
        dir_path: The path to a directory containing log files
        log_extension: The extension of the log files
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply(zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided
        ignore_metrics: Any metrics within the log files which will not be visualized
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None
    """
    assert os.path.isdir(dir_path), "provided path is not a directory"

    file_paths = []
    for file_name in os.listdir(dir_path):
        file_name = os.fsdecode(file_name)
        file_path = os.path.join(dir_path, file_name)

        if file_name.endswith(log_extension):
            file_paths.append(file_path)

    parse_files(file_paths, log_extension, smooth_factor, save, save_path, ignore_metrics, share_legend, pretty_names)


class SaveAction(argparse.Action):
    """
    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
        is invoked directly during argument parsing.
    """

    def __init__(self, option_strings, dest, nargs='?', **kwargs):
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super(SaveAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))


if __name__ == '__main__':
    parser_instance = argparse.ArgumentParser(description='Generates comparison graphs amongst one or more log files',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_instance.add_argument('log_dir', metavar='<Log Dir>', type=str,
                                 help="The path to a folder containing one or more log files")
    parser_instance.add_argument('--extension', metavar='E', type=str, help="The file type / extension of your logs",
                                 default=".txt")
    parser_instance.add_argument('--ignore', metavar='I', type=str, nargs='+',
                                 help="The names of metrics to ignore though they may be present in the log files")
    parser_instance.add_argument('--smooth', metavar='<float>', type=float,
                                 help="The amount of gaussian smoothing to apply (zero for no smoothing)",
                                 default=1)
    parser_instance.add_argument('--pretty_names', help="Clean up the metric names for display", action='store_true')

    legend_group = parser_instance.add_argument_group('legend arguments')
    legend_x_group = legend_group.add_mutually_exclusive_group(required=False)
    legend_x_group.add_argument('--common_legend', dest='share_legend', help="Generate one legend total",
                                action='store_true', default=True)
    legend_x_group.add_argument('--split_legend', dest='share_legend', help="Generate one legend per graph",
                                action='store_false', default=False)

    save_group = parser_instance.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument('--save', nargs='?', metavar='<Save Dir>',
                              help="Save the output image. May be accompanied by a directory into which the \
                                file is saved. If no output directory is specified, the log directory will be used",
                              dest='save', action=SaveAction, default=False)
    save_x_group.add_argument('--display', dest='save', action='store_false',
                              help="Render the image to the UI (rather than saving it)", default=True)
    save_x_group.set_defaults(save_dir=None)
    args = vars(parser_instance.parse_args())

    parse_folder(args['log_dir'], args['extension'], args['smooth'], args['save'], args['save_dir'], args['ignore'],
                 args['share_legend'], args['pretty_names'])
