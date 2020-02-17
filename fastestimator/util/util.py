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
from collections import defaultdict
from contextlib import ContextDecorator
from itertools import chain

import numpy as np
import PIL
# noinspection PyPackageRequirements
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.distribute.values import DistributedValues


def load_image(file_path, strip_alpha=False, channels=3):
    """
    Args:
        file_path: The path to an image file
        strip_alpha: True to convert an RGBA image to RGB
        channels: How many channels should the image have (0,1,3)
    Returns:
        The image loaded into memory and scaled to a range of [-1, 1]
    """
    # noinspection PyUnresolvedReferences
    img = PIL.Image.open(file_path)
    if strip_alpha and img.mode == "RGBA":
        # noinspection PyUnresolvedReferences
        background = PIL.Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        img = background
    if channels == 0 or channels == 1:
        img = img.convert("L")
    if channels == 3:
        img = img.convert("RGB")
    img = np.asarray(img) / 127.5 - 1.0
    if channels == 1:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img


def load_dict(dict_path, array_key=False):
    """
    Args:
        dict_path: The path to a json dictionary
        array_key: If true the parser will consider the first element in a sublist at the key {_:[K,...,V],...}.
                    Otherwise it will parse as {K:V,...} or {K:[...,V],...}
    Returns:
        A dictionary corresponding to the info from the file. If the file was formatted with arrays as the values for a
        key, the last element of the array is used as the value for the key in the parsed dictionary
    """
    parsed = None
    if dict_path is not None:
        with open(dict_path) as open_file:
            parsed = json.load(open_file)
            for key in list(parsed.keys()):
                entry = parsed[key]
                if isinstance(entry, list):
                    if array_key:
                        val = parsed.pop(key)
                        parsed[val[0]] = val[-1]
                    else:
                        parsed[key] = parsed[key][-1]
    return parsed


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


def convert_tf_dtype(datatype):
    """Return the tensorflow datatype from string.

    Args:
        datatype: String of datatype

    Returns:
        Tensor data type
    """
    datatype_map = {
        "string": tf.string,
        "int8": tf.int8,
        "uint8": tf.uint8,
        "int16": tf.int16,
        "uint16": tf.uint16,
        "int32": tf.int32,
        "uint32": tf.uint32,
        "int64": tf.int64,
        "uint64": tf.uint64,
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64
    }
    return datatype_map[datatype]


def convert_np_dtype(datatype):
    """Return the numpy datatype from string.

    Args:
        datatype: String of datatype

    Returns:
        numpy data type
    """
    datatype_map = {
        "bool": np.bool,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "string": np.str
    }
    return datatype_map[datatype]


def strip_suffix(target, suffix):
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


def prettify_metric_name(metric):
    """Add spaces to camel case words, then swap _ for space, and capitalize each word

    Args:
        metric: A string to be formatted

    Returns:
        The formatted version of 'metric'

    """
    return string.capwords(re.sub("([a-z])([A-Z])", r"\g<1> \g<2>", metric).replace("_", " "))


def remove_blacklist_keys(dic, blacklist):
    """A function which removes the blacklisted elements from a dictionary

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


def is_number(s):  # pylint: disable=invalid-name
    """Check if a given string can be converted into a number.

    Args:
        s: A string

    Returns:
        True iff the string represents a number

    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def decode_predictions(predictions, top=3, dictionary=None):
    """
    Args:
        predictions: A batched numpy array of class prediction scores (Batch X Predictions)
        top: How many of the highest predictions to capture
        dictionary: {"<class_idx>" -> "<class_name>"}
    Returns:
        A right-justified newline-separated array of the top classes and their associated probabilities.
        There is one entry in the results array per batch in the input
    """
    results = []
    for prediction in predictions:
        top_indices = prediction.argsort()[-top:][::-1]
        if dictionary is None:
            result = ["Class {}: {:.4f}".format(i, prediction[i]) for i in top_indices]
        else:
            result = [
                "{}: {:.4f}".format(dictionary.get(i, dictionary.get(str(i), "Class {}".format(i))), prediction[i])
                for i in top_indices
            ]
        max_width = len(max(result, key=len))
        result = str.join("\n", [s.rjust(max_width) for s in result])
        results.append(result)
    return results


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


class NonContext(object):
    """A class which is used for nothing.
    """
    def __enter__(self):
        pass

    def __exit__(self, *exc):
        pass


def get_num_devices():
    """Return number of devices.

    Returns:
        int: Number of GPUs available. Returns 1 if no GPU is found.
    """
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return max(1, len(gpu_list))


def flatten_list(input_list):
    """Return a flattened list.

    Args:
        input_list (list): A list that might be nested.

    Returns:
        list: A list that is not nested.

    """
    for idx, ele in enumerate(input_list):
        input_list[idx] = to_list(ele)
    output_list = list(chain.from_iterable(input_list))
    return output_list


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


def to_set(data):
    """Convert data to a set.

    Args:
        data: Input data, with or without a python container.

    Returns:
        list: Replace python container with set or make input a set.

    """
    if not isinstance(data, set):
        if isinstance(data, (tuple, list)):
            data = set(data)
        else:
            data = {data}
    return data


def per_replica_to_global(data):
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
            return tf.stack(data.values)
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


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
