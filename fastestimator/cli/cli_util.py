import argparse
import os
from typing import List, Dict, Any

from fastestimator.util.util import parse_string_to_python


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
