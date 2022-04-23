#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import argparse
import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Optional, Sequence, Union


class SaveAction(argparse.Action):
    """A customized save action for use with argparse.

    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
    is invoked directly during argument parsing.

    This class is intentionally not @traceable.

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


def parse_string_to_python(val: str) -> Any:
    """Convert a string into a python object.

    ```python
    x = fe.util.parse_string_to_python("5")  # 5
    x = fe.util.parse_string_to_python("[5, 4, 0.3]")  # [5, 4, 0.3]
    x = fe.util.parse_string_to_python("{'a':5, 'b':7}")  # {'a':5, 'b':7}
    ```

    Args:
        val: An input string.

    Returns:
        A python object version of the input string.
    """
    if val is None or not val:
        return ""
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
