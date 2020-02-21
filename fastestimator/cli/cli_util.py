import os
import re

from fastestimator.dataset import UnlabeledDirDataset
from fastestimator.summary import Summary
from fastestimator.summary.logs import visualize_logs
from fastestimator.util.util import strip_suffix, parse_string_to_python
from typing import List, Optional, Set, Dict, Any
import argparse


class SaveAction(argparse.Action):
    """
    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
    is invoked directly during argument parsing.
    """
    def __init__(self, option_strings, dest, nargs='?', **kwargs):
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))


def parse_cli_to_dictionary(input_list: List[str]) -> Dict[str, Any]:
    """
    Args:
        input_list: A list of input strings from a cli

    Returns:
        A dictionary constructed from the input list, with values converted to python objects where applicable
    """
    result = {}
    if input_list is None:
        return result
    key = ""
    val = ""
    idx = 0
    while idx < len(input_list):
        if input_list[idx].startswith("--"):
            if len(key) > 0:
                result[key] = parse_string_to_python(val)
            val = ""
            key = input_list[idx].strip('--')
        else:
            val += input_list[idx]
        idx += 1
    if len(key) > 0:
        result[key] = parse_string_to_python(val)
    return result


def _parse_file(file_path: str, file_extension: str) -> Summary:
    """ A function which will parse log files into a dictionary of metrics

    Args:
        file_path: The path to a log file
        file_extension: The extension of the log file
    Returns:
        An experiment summarizing the given log file
    """
    # TODO: need to handle multi-line output like confusion matrix
    experiment = Summary(strip_suffix(os.path.split(file_path)[1].strip(), file_extension))
    with open(file_path) as file:
        for line in file:
            mode = None
            if line.startswith("FastEstimator-Train"):
                mode = "train"
            elif line.startswith("FastEstimator-Eval"):
                mode = "eval"
            if mode is None:
                continue
            parsed_line = re.findall(r"([^:^;\s]+):[\s]*([-]?[0-9]+[.]?[0-9]*);", line)
            step = parsed_line[0]
            assert step[0] == "step", \
                "Log file (%s) seems to be missing step information, or step is not listed first" % file
            for metric in parsed_line[1:]:
                experiment.history[mode][metric[0]].update({int(step[1]): float(metric[1])})
    return experiment


def parse_log_files(file_paths: List[str],
                    log_extension: Optional[str] = '.txt',
                    smooth_factor: float = 0,
                    save: bool = False,
                    save_path: Optional[str] = None,
                    ignore_metrics: Optional[Set[str]] = None,
                    share_legend: bool = True,
                    pretty_names: bool = False):
    """A function which will iterate through the given log file paths, parse them to extract metrics, remove any
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
    """
    if file_paths is None or len(file_paths) < 1:
        raise AssertionError("must provide at least one log file")
    if save and save_path is None:
        save_path = file_paths[0]

    experiments = []
    for file_path in file_paths:
        experiments.append(_parse_file(file_path, log_extension))
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
                  pretty_names: bool = False):
    """ A function which will gather all log files within a given folder and pass them along for visualization

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
    """
    loader = UnlabeledDirDataset(root_dir=dir_path, file_extension=log_extension, recursive_search=recursive_search)
    file_paths = list(map(lambda d: d['x'], loader.data.values()))

    parse_log_files(file_paths,
                    log_extension,
                    smooth_factor,
                    save,
                    save_path,
                    ignore_metrics,
                    share_legend,
                    pretty_names)
