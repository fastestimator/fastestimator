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
"""Utilities for FastEstimator."""
import json
import re
import string
import subprocess
from ast import literal_eval
from functools import reduce
from math import gcd
from typing import Any, List, Set, Optional

from pyfiglet import Figlet


def parse_string_to_python(val: str) -> Any:
    """
    Args:
        val: An input string
    Returns:
        A python object version of the input string
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


def to_list(data: Any) -> List[Any]:
    """Convert data to a list. A single None value will be converted to the empty list.
    Args:
        data: Input data, with or without a python container.
    Returns:
        list: Replace python container with list or make input a list.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        if isinstance(data, (tuple, set)):
            data = list(data)
        else:
            data = [data]
    return data


def to_set(data: Any) -> Set[Any]:
    """Convert data to a set. A single None value will be converted to the empty set.
    Args:
        data: Input data, with or without a python container.
    Returns:
        list: Replace python container with set or make input a set.
    """
    if data is None:
        return set()
    if not isinstance(data, set):
        if isinstance(data, (tuple, list)):
            data = set(data)
        else:
            data = {data}
    return data


class NonContext(object):
    """A class which is used for nothing.
    """
    def __enter__(self):
        pass

    def __exit__(self, *exc):
        pass


def draw():
    print(Figlet(font="slant").renderText("FastEstimator"))


def get_num_devices() -> int:
    try:
        result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = [line.split() for line in result.splitlines() if line.startswith("Attached GPUs")]
        num_devices = int(lines[0][-1])
    except:
        num_devices = 1
    return num_devices


def lcms(*numbers):
    def lcm(a, b):
        return int(a * b / gcd(a, b))

    return reduce(lcm, numbers)


def prettify_metric_name(metric: str) -> str:
    """Add spaces to camel case words, then swap _ for space, and capitalize each word

    Args:
        metric: A string to be formatted
    Returns:
        The formatted version of 'metric'
    """
    return string.capwords(re.sub("([a-z])([A-Z])", r"\g<1> \g<2>", metric).replace("_", " "))


def strip_suffix(target: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Remove the given suffix from the target if it is present there

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
