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
from typing import List, Optional, Set

from fastestimator.dataset.dir_dataset import DirDataset
from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.summary.summary import Summary
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
            parsed_line = re.findall(r"([^:;]+):[\s]*([-]?[0-9]+[.]?[0-9]*(e[-]?[0-9]+[.]?[0-9]*)?);", line)
            step = parsed_line[0]
            assert step[0].strip() == "step", \
                "Log file (%s) seems to be missing step information, or step is not listed first" % file
            for metric in parsed_line[1:]:
                experiment.history[mode][metric[0].strip()].update({int(step[1]): float(metric[1])})
    return experiment


def parse_log_files(file_paths: List[str],
                    log_extension: Optional[str] = '.txt',
                    smooth_factor: float = 0,
                    save: bool = False,
                    save_path: Optional[str] = None,
                    ignore_metrics: Optional[Set[str]] = None,
                    share_legend: bool = True,
                    pretty_names: bool = False) -> None:
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
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).

    Raises:
        AssertionError: If no log files are provided.
    """
    if file_paths is None or len(file_paths) < 1:
        raise AssertionError("must provide at least one log file")
    if save and save_path is None:
        save_path = file_paths[0]

    experiments = []
    for file_path in file_paths:
        experiments.append(parse_log_file(file_path, log_extension))
    visualize_logs(experiments,
                   save_path=save_path,
                   smooth_factor=smooth_factor,
                   share_legend=share_legend,
                   pretty_names=pretty_names,
                   ignore_metrics=ignore_metrics)


def parse_log_dir(dir_path: str,
                  log_extension: str = '.txt',
                  recursive_search: bool = False,
                  smooth_factor: float = 1,
                  save: bool = False,
                  save_path: Optional[str] = None,
                  ignore_metrics: Optional[Set[str]] = None,
                  share_legend: bool = True,
                  pretty_names: bool = False) -> None:
    """A function which will gather all log files within a given folder and pass them along for visualization.

    Args:
        dir_path: The path to a directory containing log files.
        log_extension: The extension of the log files.
        recursive_search: Whether to recursively search sub-directories for log files.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        save: Whether to save (True) or display (False) the generated graph.
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided.
        ignore_metrics: Any metrics within the log files which will not be visualized.
        share_legend: Whether to have one legend across all graphs (True) or one legend per graph (False).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
    """
    loader = DirDataset(root_dir=dir_path, file_extension=log_extension, recursive_search=recursive_search)
    file_paths = list(map(lambda d: d['x'], loader.data.values()))

    parse_log_files(file_paths,
                    log_extension,
                    smooth_factor,
                    save,
                    save_path,
                    ignore_metrics,
                    share_legend,
                    pretty_names)
