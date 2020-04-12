import argparse
import os
from typing import List, Dict, Any, Sequence, Union, Optional

from fastestimator.util.util import parse_string_to_python


class SaveAction(argparse.Action):
    """A customized save action for use with argparse.

    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
    is invoked directly during argument parsing.

    Args:
        option_strings: A list of command-line option strings which should be associated with this action.
        dest: The name of the attribute to hold the created object(s).
        nargs: The number of command line arguments to be consumed.
        **kwargs: Pass-through keyword arguments.
    """
    def __init__(self,
                 option_strings: Sequence[str],
                 dest: str,
                 nargs: Union[int, str, None] = '?',
                 **kwargs: Dict[str, Any]) -> None:
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self,
                 parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace,
                 values: Optional[str],
                 option_string: Optional[str] = None) -> None:
        """Invokes the save action, writing two values into the namespace.

        Args:
            parser: The active argument parser (ignored by this implementation).
            namespace: The current namespace to be written to.
            values: The value to write into the namespace.
            option_string: An option_string (ignored by this implementation).
        """
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))


def parse_cli_to_dictionary(input_list: List[str]) -> Dict[str, Any]:
    """Convert a list of strings into a dictionary with python objects as values.

    ```python
    a = parse_cli_to_dictionary(["--epochs", "5", "--test", "this", "--lr", "0.74"]) 
    # {'epochs': 5, 'test': 'this', 'lr': 0.74}
    ```

    Args:
        input_list: A list of input strings from the cli.

    Returns:
        A dictionary constructed from the `input_list`, with values converted to python objects where applicable.
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
