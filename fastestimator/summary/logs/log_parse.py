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
import os
import re
from collections import defaultdict
from typing import List, Optional, Set

from fastestimator.dataset.dir_dataset import DirDataset
from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.summary.summary import Summary, ValWithError, average_summaries
from fastestimator.util.util import strip_suffix


def parse_log_file(file_path: str, file_extension: str) -> Summary:
    """A function which will parse log files into a dictionary of metrics.

    Args:
        file_path: The path to a log file.
        file_extension: The extension of the log file.
    Returns:
        An experiment summarizing the given log file.
    """
    # TODO: need to handle multi-line output like confusion matrix
    experiment = Summary(strip_suffix(os.path.split(file_path)[1].strip(), file_extension))
    last_step = 0
    last_epoch = 0
    with open(file_path) as file:
        for line in file:
            mode = None
            if line.startswith("FastEstimator-Train") or line.startswith("FastEstimator-Finish"):
                mode = "train"
            elif line.startswith("FastEstimator-Eval"):
                mode = "eval"
            elif line.startswith("FastEstimator-Test"):
                mode = "test"
            if mode is None:
                continue
            num = r"([-]?[0-9]+[.]?[0-9]*(e[-]?[0-9]+[.]?[0-9]*)?)"
            parsed_line = re.findall(r"([^:;]+):[\s]*(" + num + r"|None|\(" + num + ", " + num + ", " + num + r"\));", line)
            step = parsed_line[0]
            assert step[0].strip() == "step", \
                "Log file (%s) seems to be missing step information, or step is not listed first" % file
            step = step[1]
            adjust_epoch = False
            if step == 'None':
                # This might happen if someone runs the test mode from the cli
                step = last_step
                # If the test mode was just guessing its epoch, use the prior epoch instead
                adjust_epoch = mode == 'test'
            else:
                step = int(step)
                last_step = step
            for metric in parsed_line[1:]:
                if metric[4]:
                    val = ValWithError(float(metric[4]), float(metric[6]), float(metric[8]))
                else:
                    val = metric[1]
                    if val == 'None':
                        continue
                    val = float(val)
                key = metric[0].strip()
                if key == 'epoch':
                    if adjust_epoch:
                        val = last_epoch
                    else:
                        last_epoch = val
                experiment.history[mode][key].update({step: val})
    return experiment


def parse_log_files(file_paths: List[str],
                    log_extension: Optional[str] = '.txt',
                    smooth_factor: float = 0,
                    save: bool = False,
                    save_path: Optional[str] = None,
                    ignore_metrics: Optional[Set[str]] = None,
                    include_metrics: Optional[Set[str]] = None,
                    share_legend: bool = True,
                    pretty_names: bool = False,
                    group_by: Optional[str] = None) -> None:
    """Parse one or more log files for graphing.

    This function which will iterate through the given log file paths, parse them to extract metrics, remove any
    metrics which are blacklisted, and then pass the necessary information on the graphing function.

    Args:
        file_paths: A list of paths to various log files.
        log_extension: The extension of the log files.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        save: Whether to save (True) or display (False) the generated graph.
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided.
        ignore_metrics: Any metrics within the log files which will not be visualized.
        include_metrics: A whitelist of metric keys (None whitelists all keys).
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        group_by: Combine multiple log files by a regex to visualize their mean+-stddev. For example, to group together
            files like [a_1.txt, a_2.txt] vs [b_1.txt, b_2.txt] you can use: r'(.*)_[\d]+\.txt'.

    Raises:
        AssertionError: If no log files are provided.
        ValueError: If a log file does not match the `group_by` regex pattern.
    """
    if file_paths is None or len(file_paths) < 1:
        raise AssertionError("must provide at least one log file")
    if save and save_path is None:
        save_path = file_paths[0]

    groups = defaultdict(list)  # {group_name: [experiment(s)]}
    for path in file_paths:
        experiment = parse_log_file(path, log_extension)
        try:
            key = (re.findall(group_by, os.path.split(path)[1]))[0] if group_by else experiment.name
        except IndexError:
            raise ValueError(f"The log {os.path.split(path)[1]} did not match the given regex pattern: {group_by}")
        groups[key].append(experiment)
    experiments = [average_summaries(name, exps) for name, exps in groups.items()]

    visualize_logs(experiments,
                   save_path=save_path,
                   smooth_factor=smooth_factor,
                   share_legend=share_legend,
                   pretty_names=pretty_names,
                   ignore_metrics=ignore_metrics,
                   include_metrics=include_metrics)


def parse_log_dir(dir_path: str,
                  log_extension: str = '.txt',
                  recursive_search: bool = False,
                  smooth_factor: float = 1,
                  save: bool = False,
                  save_path: Optional[str] = None,
                  ignore_metrics: Optional[Set[str]] = None,
                  include_metrics: Optional[Set[str]] = None,
                  share_legend: bool = True,
                  pretty_names: bool = False,
                  group_by: Optional[str] = None) -> None:
    """A function which will gather all log files within a given folder and pass them along for visualization.

    Args:
        dir_path: The path to a directory containing log files.
        log_extension: The extension of the log files.
        recursive_search: Whether to recursively search sub-directories for log files.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        save: Whether to save (True) or display (False) the generated graph.
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided.
        ignore_metrics: Any metrics within the log files which will not be visualized.
        include_metrics: A whitelist of metric keys (None whitelists all keys).
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        group_by: Combine multiple log files by a regex to visualize their mean+-stddev. For example, to group together
            files like [a_1.txt, a_2.txt] vs [b_1.txt, b_2.txt] you can use: r'(.*)_[\d]+\.txt'.
    """
    loader = DirDataset(root_dir=dir_path, file_extension=log_extension, recursive_search=recursive_search)
    file_paths = list(map(lambda d: d['x'], loader.data.values()))

    parse_log_files(file_paths,
                    log_extension,
                    smooth_factor,
                    save,
                    save_path,
                    ignore_metrics,
                    include_metrics,
                    share_legend,
                    pretty_names,
                    group_by)
