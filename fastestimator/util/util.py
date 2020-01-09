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
import subprocess
from ast import literal_eval

from pyfiglet import Figlet


def parse_string_to_python(val):
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


def to_list(data):
    """Convert data to a list.
    Args:
        data: Input data, with or without a python container.
    Returns:
        list: Replace python container with list or make input a list.
    """
    if not isinstance(data, list):
        if isinstance(data, (tuple, set)):
            data = list(data)
        else:
            data = [data]
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


def get_num_devices():
    try:
        result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = [line.split() for line in result.splitlines() if line.startswith("Attached GPUs")]
        num_devices = int(lines[0][-1])
    except:
        num_devices = 1
    return num_devices
