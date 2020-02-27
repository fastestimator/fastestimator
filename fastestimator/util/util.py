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
import sys
import time
from ast import literal_eval
from contextlib import ContextDecorator
from functools import reduce
from math import gcd
from typing import Any, List, Optional, Set

import tensorflow as tf
from pyfiglet import Figlet
from tensorflow.python.distribute.values import DistributedValues


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


class Suppressor(object):
    """A class which can be used to silence output of function calls.
    Example: ::
        with Suppressor():
            func(args)
    """
    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # pylint: enable=attribute-defined-outside-init
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type is not None:
            raise

    def write(self, dummy):  # pylint: disable=missing-docstring
        pass


class Timer(ContextDecorator):
    """
    A class that can be used to time things: ::
        with Timer():
            func(args)
        @Timer()
        def func(args)
    """
    def __init__(self, name="Task"):
        self.name = name
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        tf.print("{} took {} seconds".format(self.name, self.interval))


def draw():
    print(Figlet(font="slant").renderText("FastEstimator"))


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

def is_number(number: str) -> bool:  # pylint: disable=invalid-name
    """Check if a given string can be converted into a number.

    Args:
        number: A number stored as string

    Returns:
        True if the string represents a number

    """
    try:
        float(number)
        return True
    except (ValueError, TypeError):
        return False

def per_replica_to_global(data: Any) -> Any:
    """Combine data from "per-replica" values.
    For multi-GPU training, data are distributed using `tf.distribute.Strategy.experimental_distribute_dataset`. This
    method collects data from all replicas and combine them into one.
    Args:
        data: Distributed data.
    Returns:
        obj: Combined data from all replicas.
    """
    if isinstance(data, DistributedValues):
        if data.values[0].shape.rank == 0:
            return tf.reduce_mean(data.values)
        else:
            return tf.concat(data.values, axis=0)
    if isinstance(data, dict):
        result = {}
        for key, val in data.items():
            result[key] = per_replica_to_global(val)
        return result
    if isinstance(data, list):
        return [per_replica_to_global(val) for val in data]
    if isinstance(data, tuple):
        return tuple([per_replica_to_global(val) for val in data])
    if isinstance(data, set):
        return set([per_replica_to_global(val) for val in data])
