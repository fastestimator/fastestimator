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
import os
import re
import string
import sys
import time
from ast import literal_eval
from contextlib import ContextDecorator
from typing import Any, Callable, Dict, KeysView, List, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pyfiglet import Figlet

STRING_TO_TORCH_DTYPE = {
    None: None,
    'float32': torch.float32,
    'float': torch.float,
    'float64': torch.float64,
    'double': torch.double,
    'float16': torch.float16,
    'half': torch.half,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.short,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.long,
    'bool': torch.bool
}

STRING_TO_TF_DTYPE = {
    None: None,
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

TENSOR_TO_NP_DTYPE = {
    # Abstract types like 'float' and 'long' are intentionally not included here since they are never actually a
    # tensor's dtype and they interfere with the finer-grained keys (torch.float intercepts torch.float32, for example)
    None: None,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.float16: np.float16,
    tf.uint8: np.uint8,
    tf.int8: np.int8,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.bool: np.bool,
    np.dtype('float32'): np.float32,
    np.dtype('float64'): np.float64,
    np.dtype('float16'): np.float16,
    np.dtype('uint8'): np.uint8,
    np.dtype('int8'): np.int8,
    np.dtype('int16'): np.int16,
    np.dtype('int32'): np.int32,
    np.dtype('int64'): np.int64,
    np.dtype('bool'): np.bool,
}

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


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


def to_list(data: Any) -> List[Any]:
    """Convert data to a list. A single None value will be converted to the empty list.

    ```python
    x = fe.util.to_list(None)  # []
    x = fe.util.to_list([None])  # [None]
    x = fe.util.to_list(7)  # [7]
    x = fe.util.to_list([7, 8])  # [7,8]
    x = fe.util.to_list({7})  # [7]
    x = fe.util.to_list((7))  # [7]
    x = fe.util.to_list({'a': 7})  # [{'a': 7}]
    ```

    Args:
        data: Input data, within or without a python container.

    Returns:
        The input `data` but inside a list instead of whatever other container type used to hold it.
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

    ```python
    x = fe.util.to_set(None)  # set()
    x = fe.util.to_set([None])  # {None}
    x = fe.util.to_set(7)  # {7}
    x = fe.util.to_set([7, 8])  # {7,8}
    x = fe.util.to_set({7})  # {7}
    x = fe.util.to_set((7))  # {7}
    ```

    Args:
        data: Input data, within or without a python container. The `data` must be hashable.

    Returns:
        The input `data` but inside a set instead of whatever other container type used to hold it.
    """
    if data is None:
        return set()
    if not isinstance(data, set):
        if isinstance(data, (tuple, list, KeysView)):
            data = set(data)
        else:
            data = {data}
    return data


def param_to_range(
        data: Union[int, float, Tuple[int, int], Tuple[float, float]]) -> Union[Tuple[int, int], Tuple[float, float]]:
    """Convert a single int or float value to a tuple signifying a range.

    ```python
    x = fe.util.param_to_tuple(7)  # (-7, 7)
    x = fe.util.param_to_tuple([7, 8])  # (7,8))
    x = fe.util.param_to_tuple((3.1, 4.3))  # (3.1, 4.3)
    x = fe.util.to_set((-3.2))  # (-3.2, 3.2)
    ```

    Args:
        data: Input data.

    Returns:
        The input `data` but in tuple form for a range.
    """
    if isinstance(data, (int, float)):
        if data > 0:
            data = -data, data
        else:
            data = data, -data
    elif isinstance(data, (list, tuple)):
        data = tuple(data)

    return data


class NonContext(object):
    """A class which is used to make nothing unusual happen.

    This class is intentionally not @traceable.

    ```python
    a = 5
    with fe.util.NonContext():
        a = a + 37
    print(a)  # 42
    ```
    """
    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        pass


class Suppressor(object):
    """A class which can be used to silence output of function calls.

    This class is intentionally not @traceable.

    ```python
    x = lambda: print("hello")
    x()  # "hello"
    with fe.util.Suppressor():
        x()  #
    x()  # "hello"
    ```
    """
    def __enter__(self) -> None:
        # This is not necessary to block printing, but lets the system know what's happening
        self.py_reals = [sys.stdout, sys.stderr]
        sys.stdout = sys.stderr = self
        # This part does the heavy lifting
        self.fakes = [os.open(os.devnull, os.O_RDWR), os.open(os.devnull, os.O_RDWR)]
        self.reals = [os.dup(1), os.dup(2)]  # [stdout, stderr]
        os.dup2(self.fakes[0], 1)
        os.dup2(self.fakes[1], 2)

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        os.dup2(self.reals[0], 1)
        os.dup2(self.reals[1], 2)
        for fd in self.fakes + self.reals:
            os.close(fd)
        # Set the python pointers back too
        sys.stdout, sys.stderr = self.py_reals[0], self.py_reals[1]

    def write(self, dummy: str) -> None:
        """A function which is invoked during print calls.

        Args:
            dummy: The string which wanted to be printed.
        """
        pass


class LogSplicer:
    """A class to send stdout information into a file before passing it along to the normal stdout.

    Args:
        log_path: The path/filename into which to append the current stdout.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.stdout = None
        self.log_file = None

    def __enter__(self) -> None:
        self.log_file = open(self.log_path, 'a')
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        sys.stdout = self.stdout
        self.log_file.close()

    def write(self, output: str) -> None:
        self.log_file.write(output)
        self.stdout.write(output)

    def flush(self) -> None:
        self.stdout.flush()
        self.log_file.flush()


class Timer(ContextDecorator):
    """A class that can be used to time things.

    This class is intentionally not @traceable.

    ```python
    x = lambda: list(map(lambda i: i + i/2, list(range(int(1e6)))))
    with fe.util.Timer():
        x()  # Task took 0.1639 seconds
    @fe.util.Timer("T2")
    def func():
        return x()
    func()  # T2 took 0.14819 seconds
    ```
    """
    def __init__(self, name="Task") -> None:
        self.name = name
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        tf.print("{} took {} seconds".format(self.name, self.interval))


def draw() -> None:
    """Print our name.
    """
    print(Figlet(font="slant").renderText("FastEstimator"))


def prettify_metric_name(metric: str) -> str:
    """Add spaces to camel case words, then swap _ for space, and capitalize each word.

    ```python
    x = fe.util.prettify_metric_name("myUgly_loss")  # "My Ugly Loss"
    ```

    Args:
        metric: A string to be formatted.

    Returns:
        The formatted version of 'metric'.
    """
    return string.capwords(re.sub("([a-z])([A-Z])", r"\g<1> \g<2>", metric).replace("_", " "))


def strip_suffix(target: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Remove the given `suffix` from the `target` if it is present there.

    ```python
    x = fe.util.strip_suffix("astring.json", ".json")  # "astring"
    x = fe.util.strip_suffix("astring.json", ".yson")  # "astring.json"
    ```

    Args:
        target: A string to be formatted.
        suffix: A string to be removed from `target`.

    Returns:
        The formatted version of `target`.
    """
    if suffix is None or target is None:
        return target
    s_len = len(suffix)
    if target[-s_len:] == suffix:
        return target[:-s_len]
    return target


def strip_prefix(target: Optional[str], prefix: Optional[str]) -> Optional[str]:
    """Remove the given `prefix` from the `target` if it is present there.

    ```python
    x = fe.util.strip_prefix("astring.json", "ast")  # "ring.json"
    x = fe.util.strip_prefix("astring.json", "asa")  # "astring.json"
    ```

    Args:
        target: A string to be formatted.
        prefix: A string to be removed from `target`.

    Returns:
        The formatted version of `target`.
    """
    if prefix is None or target is None:
        return target
    s_len = len(prefix)
    if target[:s_len] == prefix:
        return target[s_len:]
    return target


def get_type(obj: Any) -> str:
    """A function to try and infer the types of data within containers.

    ```python
    x = fe.util.get_type(np.ones((10, 10), dtype='int32'))  # "int32"
    x = fe.util.get_type(tf.ones((10, 10), dtype='float16'))  # "<dtype: 'float16'>"
    x = fe.util.get_type(torch.ones((10, 10)).type(torch.float))  # "torch.float32"
    x = fe.util.get_type([np.ones((10,10)) for i in range(4)])  # "List[float64]"
    x = fe.util.get_type(27)  # "int"
    ```

    For container to look into its element's type, its type needs to be either list or tuple, and the return string will
    be List[...]. All container elements need to have the same data type becuase it will only check its first element.

    ```python
    x = fe.util.get_type({"a":1, "b":2})  # "dict"
    x = fe.util.get_type([1, "a"]) # "List[int]"
    x = fe.util.get_type([[[1]]]) # "List[List[List[int]]]"
    ```

    Args:
        obj: Data which may be wrapped in some kind of container.

    Returns:
        A string representation of the data type of the `obj`.
    """
    if hasattr(obj, "dtype"):
        result = str(obj.dtype)
    elif isinstance(obj, (List, Tuple)):
        if len(obj) > 0:
            result = "List[{}]".format(get_type(obj[0]))
        else:
            result = strip_suffix(strip_prefix(str(type(obj)), "<class '"), "'>")
    else:
        result = strip_suffix(strip_prefix(str(type(obj)), "<class '"), "'>")
    return result


def get_shape(obj: Any) -> List[Optional[int]]:
    """A function to find the shapes of an object or sequence of objects.

    Lists or Tuples will assume that the zeroth dimension is ragged (shape==None). If entries in the list have
    mismatched ranks, then only the list dimension will be considered as part of the shape. If all ranks are equal, an
    attempt will be made to determine which of the interior dimensions are ragged.

    ```python
    x = fe.util.get_shape(np.ones((12,22,11)))  # [12, 22, 11]
    x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5))])  # [None]
    x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5, 4))])  # [None, None, None, None]
    x = fe.util.get_shape([np.ones((12,22,11)), np.ones((12, 22, 4))])  # [None, 12, 22, None]
    x = fe.util.get_shape({"a": np.ones((12,22,11))})  # []
    ```

    Args:
        obj: Data to infer the shape of.

    Returns:
        A list representing the shape of the data.
    """
    if hasattr(obj, "shape"):
        result = list(obj.shape)
    elif isinstance(obj, (List, Tuple)):
        shapes = [get_shape(ob) for ob in obj]
        result = [None]
        if shapes:
            rank = len(shapes[0])
            if any((len(shape) != rank for shape in shapes)):
                return result
            result.extend(shapes[0])
            for shape in shapes[1:]:
                for idx, dim in enumerate(shape):
                    if result[idx + 1] != dim:
                        result[idx + 1] = None
    else:
        result = []
    return result


def parse_modes(modes: Set[str]) -> Set[str]:
    """A function to determine which modes to run on based on a set of modes potentially containing blacklist values.

    ```python
    m = fe.util.parse_modes({"train"})  # {"train"}
    m = fe.util.parse_modes({"!train"})  # {"eval", "test", "infer"}
    m = fe.util.parse_modes({"train", "eval"})  # {"train", "eval"}
    m = fe.util.parse_modes({"!train", "!infer"})  # {"eval", "test"}
    ```

    Args:
        modes: The desired modes to run on (possibly containing blacklisted modes).

    Returns:
        The modes to run on (converted to a whitelist).

    Raises:
        AssertionError: If invalid modes are detected, or if blacklisted modes and whitelisted modes are mixed.
    """
    valid_fields = {"train", "eval", "test", "infer", "!train", "!eval", "!test", "!infer"}
    assert modes.issubset(valid_fields), "Invalid modes argument {}".format(modes - valid_fields)
    negation = set([mode.startswith("!") for mode in modes])
    assert len(negation) < 2, "cannot mix !mode with mode, found {}".format(modes)
    if True in negation:
        new_modes = {"train", "eval", "test", "infer"}
        for mode in modes:
            new_modes.discard(mode.strip("!"))
        modes = new_modes
    return modes


def pad_batch(batch: List[MutableMapping[str, np.ndarray]], pad_value: Union[float, int]) -> None:
    """A function to pad a batch of data in-place by appending to the ends of the tensors. Tensor type needs to be
    numpy array otherwise would get ignored. (tf.Tensor and torch.Tensor will cause error)

    ```python
    data = [{"x": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
    fe.util.pad_batch(data, pad_value=0)
    print(data)  # [{'x': [[1., 1.], [1., 1.], [0., 0.]], 'y': 8}, {'x': [[1., 0.], [1., 0.], [1., 0.]]), 'y': 4}]
    ```

    Args:
        batch: A list of data to be padded.
        pad_value: The value to pad with.

    Raises:
        AssertionError: If the data within the batch do not have matching rank, or have different keys
    """
    keys = batch[0].keys()
    for one_batch in batch:
        assert one_batch.keys() == keys, "data within batch must have same keys"

    for key in keys:
        shapes = [data[key].shape for data in batch if hasattr(data[key], "shape")]
        if len(set(shapes)) > 1:
            assert len(set(len(shape) for shape in shapes)) == 1, "data within batch must have same rank"
            max_shapes = tuple(np.max(np.array(shapes), axis=0))
            for data in batch:
                data[key] = pad_data(data[key], max_shapes, pad_value)


def pad_data(data: np.ndarray, target_shape: Tuple[int, ...], pad_value: Union[float, int]) -> np.ndarray:
    """Pad `data` by appending `pad_value`s along it's dimensions until the `target_shape` is reached. All entris of
    target_shape should be larger than the data.shape, and have the same rank.

    ```python
    x = np.ones((1,2))
    x = fe.util.pad_data(x, target_shape=(3, 3), pad_value = -2)  # [[1, 1, -2], [-2, -2, -2], [-2, -2, -2]]
    x = fe.util.pad_data(x, target_shape=(3, 3, 3), pad_value = -2) # error
    x = fe.util.pad_data(x, target_shape=(4, 1), pad_value = -2) # error
    ```

    Args:
        data: The data to be padded.
        target_shape: The desired shape for `data`. Should have the same rank as `data`, with each dimension being >=
            the size of the `data` dimension.
        pad_value: The value to insert into `data` if padding is required to achieve the `target_shape`.

    Returns:
        The `data`, padded to the `target_shape`.
    """
    shape_difference = np.array(target_shape) - np.array(data.shape)
    padded_shape = np.array([np.zeros_like(shape_difference), shape_difference]).T
    return np.pad(data, padded_shape, 'constant', constant_values=pad_value)


def is_number(arg: str) -> bool:
    """Check if a given string can be converted into a number.

    ```python
    x = fe.util.is_number("13.7")  # True
    x = fe.util.is_number("ae13.7")  # False
    ```

    Args:
        arg: A potentially numeric input string.

    Returns:
        True iff `arg` represents a number.
    """
    try:
        float(arg)
        return True
    except (ValueError, TypeError):
        return False


class DefaultKeyDict(dict):
    """Like collections.defaultdict but it passes the key argument to the default function.

    This class is intentionally not @traceable.

    ```python
    d = fe.util.DefaultKeyDict(default=lambda x: x+x, a=4, b=6)
    print(d["a"])  # 4
    print(d["c"])  # "cc"
    ```

    Args:
        default: A function which takes a key and returns a default value based on the key.
        **kwargs: Initial key/value pairs for the dictionary.
    """
    def __init__(self, default: Callable[[Any], Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self.factory = default

    def __missing__(self, key: Any) -> Any:
        res = self[key] = self.factory(key)
        return res


def get_num_devices():
    """Determine the number of available GPUs.

    Returns:
        The number of available GPUs, or 1 if none are found.
    """
    return max(torch.cuda.device_count(), 1)


def show_image(im: Union[np.ndarray, Tensor],
               axis: plt.Axes = None,
               fig: plt.Figure = None,
               title: Optional[str] = None,
               color_map: str = "inferno",
               stack_depth: int = 0) -> Optional[plt.Figure]:
    """Plots a given image onto an axis. The repeated invocation of this function will cause figure plot overlap.

    If `im` is 2D and the length of second dimension are 4 or 5, it will be viewed as bounding box data (x0, y0, w, h,
    <label>).

    ```python
    boxes = np.array([[0, 0, 10, 20, "apple"],
                      [10, 20, 30, 50, "dog"],
                      [40, 70, 200, 200, "cat"],
                      [0, 0, 0, 0, "not_shown"],
                      [0, 0, -10, -20, "not_shown2"]])

    img = np.zeros((150, 150))
    fig, axis = plt.subplots(1, 1)
    fe.util.show_image(img, fig=fig, axis=axis) # need to plot image first
    fe.util.show_image(boxes, fig=fig, axis=axis)
    ```

    Users can also directly plot text

    ```python
    fig, axis = plt.subplots(1, 1)
    fe.util.show_image("apple", fig=fig, axis=axis)
    ```

    Args:
        axis: The matplotlib axis to plot on, or None for a new plot.
        fig: A reference to the figure to plot on, or None if new plot.
        im: The image (width X height) / bounding box / text to display.
        title: A title for the image.
        color_map: Which colormap to use for greyscale images.
        stack_depth: Multiple images can be drawn onto the same axis. When stack depth is greater than zero, the `im`
            will be alpha blended on top of a given axis.

    Returns:
        plotted figure. It will be the same object as user have provided in the argument.
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    axis.axis('off')
    # Compute width of axis for text font size
    bbox = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    space = min(width, height)
    if not hasattr(im, 'shape') or len(im.shape) < 2:
        # text data
        im = to_number(im)
        if hasattr(im, 'shape') and len(im.shape) == 1:
            im = im[0]
        im = im.item()
        if isinstance(im, bytes):
            im = im.decode('utf8')
        text = "{}".format(im)
        axis.text(0.5,
                  0.5,
                  im,
                  ha='center',
                  transform=axis.transAxes,
                  va='center',
                  wrap=False,
                  family='monospace',
                  fontsize=min(45, space // len(text)))
    elif len(im.shape) == 2 and (im.shape[1] == 4 or im.shape[1] == 5):
        # Bounding Box Data. Should be (x0, y0, w, h, <label>)
        boxes = []
        im = to_number(im)
        color = ["m", "r", "c", "g", "y", "b"][stack_depth % 6]
        for box in im:
            # Unpack the box, which may or may not have a label
            x0 = float(box[0])
            y0 = float(box[1])
            width = float(box[2])
            height = float(box[3])
            label = None if len(box) < 5 else str(box[4])

            # Don't draw empty boxes, or invalid box
            if width <= 0 or height <= 0:
                continue
            r = Rectangle((x0, y0), width=width, height=height, fill=False, edgecolor=color, linewidth=3)
            boxes.append(r)
            if label:
                axis.text(r.get_x() + 3,
                          r.get_y() + 3,
                          label,
                          ha='left',
                          va='top',
                          color=color,
                          fontsize=max(8, min(14, width // len(label))),
                          fontweight='bold',
                          family='monospace')
        pc = PatchCollection(boxes, match_original=True)
        axis.add_collection(pc)
    else:
        if isinstance(im, torch.Tensor) and len(im.shape) > 2:
            # Move channel first to channel last
            channels = list(range(len(im.shape)))
            channels.append(channels.pop(0))
            im = im.permute(*channels)
        # image data
        im = to_number(im)
        im_max = np.max(im)
        im_min = np.min(im)
        if np.issubdtype(im.dtype, np.integer):
            # im is already in int format
            im = im.astype(np.uint8)
        elif 0 <= im_min <= im_max <= 1:  # im is [0,1]
            im = (im * 255).astype(np.uint8)
        elif -0.5 <= im_min < 0 < im_max <= 0.5:  # im is [-0.5, 0.5]
            im = ((im + 0.5) * 255).astype(np.uint8)
        elif -1 <= im_min < 0 < im_max <= 1:  # im is [-1, 1]
            im = ((im + 1) * 127.5).astype(np.uint8)
        else:  # im is in some arbitrary range, probably due to the Normalize Op
            ma = abs(np.max(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            mi = abs(np.min(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            im = (((im + mi) / (ma + mi)) * 255).astype(np.uint8)
        # matplotlib doesn't support (x,y,1) images, so convert them to (x,y)
        if len(im.shape) == 3 and im.shape[2] == 1:
            im = np.reshape(im, (im.shape[0], im.shape[1]))
        alpha = 1 if stack_depth == 0 else 0.3
        if len(im.shape) == 2:
            axis.imshow(im, cmap=plt.get_cmap(name=color_map), alpha=alpha)
        else:
            axis.imshow(im, alpha=alpha)
    if title is not None:
        axis.set_title(title, fontsize=min(20, 1 + width // len(title)), family='monospace')
    return fig


def get_batch_size(data: Dict[str, Any]) -> int:
    """Infer batch size from a batch dictionary. It will ignore all dictionary value with data type that
    doesn't have "shape" attribute.

    Args:
        data: The batch dictionary.

    Returns:
        batch size.
    """
    assert isinstance(data, dict), "data input must be a dictionary"
    batch_size = set(data[key].shape[0] for key in data if hasattr(data[key], "shape") and list(data[key].shape))
    assert len(batch_size) == 1, "invalid batch size: {}".format(batch_size)
    return batch_size.pop()


class FEID:
    """An int wrapper class that can change how it's values are printed.

    This class is intentionally not @traceable.

    Args:
        val: An integer id to be wrapped.
    """
    __slots__ = ['_val']
    _translation_dict = {}

    def __init__(self, val: int):
        self._val = val

    def __hash__(self) -> int:
        return hash(self._val)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FEID):
            return self._val == other._val
        else:
            return int.__eq__(self._val, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, FEID):
            other = other._val
        return int.__lt__(self._val, other)

    def __str__(self) -> str:
        return f"{self._translation_dict.get(self._val, self._val)}"

    def __repr__(self) -> str:
        return f"{self._translation_dict.get(self._val, self._val)}"

    @classmethod
    def set_translation_dict(cls, mapping: Dict[int, Any]) -> None:
        """Provide a lookup table to be invoked during value printing.

        Args:
            mapping: A mapping of id: printable id.
        """
        cls._translation_dict.clear()
        cls._translation_dict.update(mapping)


class Flag:
    """A mutable wrapper around a boolean.

    This class is intentionally not @traceable.

    Args:
        val: The initial value for the Flag.
    """
    __slots__ = ['_val']

    def __init__(self, val: bool = False):
        self._val = val

    def set_true(self):
        self._val = True

    def set_false(self):
        self._val = False

    def __bool__(self):
        return self._val


def to_number(data: Union[tf.Tensor, torch.Tensor, np.ndarray, int, float]) -> np.ndarray:
    """Convert an input value into a Numpy ndarray.

    This method can be used with Python and Numpy data:
    ```python
    b = fe.backend.to_number(5)  # 5 (type==np.ndarray)
    b = fe.backend.to_number(4.0)  # 4.0 (type==np.ndarray)
    n = np.array([1, 2, 3])
    b = fe.backend.to_number(n)  # [1, 2, 3] (type==np.ndarray)
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([1, 2, 3])
    b = fe.backend.to_number(t)  # [1, 2, 3] (type==np.ndarray)
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([1, 2, 3])
    b = fe.backend.to_number(p)  # [1, 2, 3] (type==np.ndarray)
    ```

    Args:
        data: The value to be converted into a np.ndarray.

    Returns:
        An ndarray corresponding to the given `data`.
    """
    if tf.is_tensor(data):
        data = data.numpy()
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach().numpy()
        else:
            data = data.numpy()
    return np.array(data)
