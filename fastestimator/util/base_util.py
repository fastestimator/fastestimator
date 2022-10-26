# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

# DO NOT IMPORT FE, TF, Torch, Numpy, Seaborn, OR Matplotlib IN THIS FILE
import colorsys
import os
import re
import string
import sys
from typing import Any, Set, KeysView, List, Union, Tuple, Optional, Type, TypeVar, Dict, Callable
# DO NOT IMPORT FE, TF, Torch, Numpy, Seaborn, OR Matplotlib IN THIS FILE
from plotly.graph_objs import Figure
# DO NOT IMPORT FE, TF, Torch, Numpy, Seaborn, OR Matplotlib IN THIS FILE

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.


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

    Args:
        allow_pyprint: Whether to allow python printing to occur still within this scope (and therefore only silence
            printing from non-python sources like c).

    ```python
    x = lambda: print("hello")
    x()  # "hello"
    with fe.util.Suppressor():
        x()  #
    x()  # "hello"
    ```
    """
    def __init__(self, allow_pyprint: bool = False):
        self.allow_pyprint = allow_pyprint

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
        if self.allow_pyprint:
            os.write(self.reals[0], dummy.encode('utf-8'))

    def flush(self) -> None:
        """A function to empty the current print buffer. No-op in this case.
        """


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

    def getvalue(self) -> str:
        return self.stdout.getvalue()


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


def check_io_names(names: List[Optional[str]]) -> List[Optional[str]]:
    forbidden_chars = {":", ";"}
    for name in names:
        assert not any(char in name for char in forbidden_chars), \
            "inputs/outputs name cannot contain characters like ':', ';', found {}".format(name)
        assert len(name) > 0, "inputs/outputs cannot be an empty string"
        assert len(name.split('|')) < 3, f"inputs/outputs cannot contain more than one '|' character, found {name}"
    return names


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


def check_ds_id(ds_ids: Set[str]) -> Set[str]:
    """A function to check whether ds_ids inputs are correct inputs.

    ds_ids should either be defined through whitelist, like {"ds1", "ds2"} or blacklist, like {"!ds1", "!ds2"}.

    ```python
    m = fe.util.parse_ds_id({"ds1"})  # {"ds1"}
    m = fe.util.parse_ds_id({"!ds1"})  # {"!ds1"}
    m = fe.util.parse_ds_id({"ds1", "ds2"})  # {"ds1", "ds2"}
    m = fe.util.parse_ds_id({"!ds1", "!ds2"})  # {"!ds1", "!ds2"}
    m = fe.util.parse_ds_id({"!ds1", "ds2"})  # Raises Assertion
    ```

    Args:
        ds_ids: The desired ds_id to run on (possibly containing blacklisted ds_ids).

    Returns:
        The ds_ids to run or to avoid.

    Raises:
        AssertionError: if blacklisted modes and whitelisted modes are mixed.
    """
    negation = set([ds_id.startswith("!") for ds_id in ds_ids])
    assert len(negation) < 2, "cannot mix !ds_id with ds_id, found {}".format(ds_ids)
    forbidden_ds_id_chars = {":", ";", "|"}
    for ds_id in ds_ids:
        assert isinstance(ds_id, str) and len(ds_id) > 0, "dataset id must be a string, found {}".format(ds_id)
        assert not any(char in ds_id for char in forbidden_ds_id_chars), \
            "dataset id should not contain forbidden characters like ':', ';', '|', found {}".format(ds_id)
    return ds_ids


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


class DefaultKeyDict(Dict[KT, VT]):
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


def in_notebook() -> bool:
    """Determine whether the code is running inside a jupyter notebook

    Returns:
        True iff the code is executing inside a Jupyter notebook
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        return False
    except (ImportError, NameError):
        return False


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


def list_files(root_dir: str,
               file_extension: Optional[str] = None,
               recursive_search: bool = True) -> List[str]:
    """Get the paths of all files in a particular root directory subject to a particular file extension.

    Args:
        root_dir: The path to the directory containing data.
        file_extension: If provided then only files ending with the file_extension will be included.
        recursive_search: Whether to search within subdirectories for files.

    Returns:
        A list of file paths found within the directory.

    Raises:
        AssertionError: If the provided path isn't a directory.
        ValueError: If the directory has an invalid structure.
    """
    paths = []
    root_dir = os.path.normpath(root_dir)
    if not os.path.isdir(root_dir):
        raise AssertionError("Provided path is not a directory")
    try:
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.startswith(".") or (file_extension is not None
                                                 and not file_name.endswith(file_extension)):
                    continue
                paths.append(os.path.join(root, file_name))
            if not recursive_search:
                break
    except StopIteration:
        raise ValueError("Invalid directory structure for DirDataset at root: {}".format(root_dir))
    return paths


def get_colors(n_colors: int,
               alpha: float = 1.0,
               as_numbers: bool = False) -> List[Union[str, Tuple[float, float, float, float]]]:
    """Get a list of colors to use in plotting.

    Args:
        n_colors: How many colors to return.
        alpha: What opacity value to use (0 to 1).
        as_numbers: Whether to return the values as a list of numbers [r,g,b,a] or as a string

    Returns:
        A list of rgba string colors.
    """
    if n_colors <= 10:
        colors = [f'rgba(1,115,178,{alpha})', f'rgba(222,143,5,{alpha})', f'rgba(2,158,115,{alpha})',
                  f'rgba(213,94,0,{alpha})', f'rgba(204,120,188,{alpha})', f'rgba(202,145,97,{alpha})',
                  f'rgba(251,175,228,{alpha})', f'rgba(148,148,148,{alpha})', f'rgba(236,225,51,{alpha})',
                  f'rgba(86,180,233,{alpha})']
    else:
        colors = [(i + 0.01) / n_colors for i in range(n_colors)]
        colors = [color - 1 if color >= 1 else color for color in colors]
        colors = [colorsys.hls_to_rgb(color, 0.6, 0.95) for color in colors]
        colors = [f'rgba({int(256*r)},{int(256*g)},{int(256*b)},{alpha})' for r, g, b in colors]
    colors = colors[:n_colors]
    if as_numbers:
        colors = [[float(x) for x in elem.strip('rgba(').strip(')').split(',')] for elem in colors]
    return colors


class FigureFE(Figure):
    @classmethod
    def from_figure(cls, fig: Figure) -> 'FigureFE':
        new_fig = FigureFE()
        new_fig.__dict__ = fig.__dict__.copy()
        return new_fig

    def _get_color(self,
                   clazz: str,
                   label: Union[int, str],
                   as_numbers: bool = False,
                   n_colors: int = 10) -> Tuple[Union[str, Tuple[float, float, float, float]], bool]:
        """A function which determines what color a plot element ought to be.

        Args:
            clazz: The class of the thing to be plotted ('mask', 'keypoint' etc.).
            label: The name of the thing to be plotted ('lung', 'patella', etc.).
            as_numbers: Whether to return the color as a tuple of rgba floats or as a rgba string.
            n_colors: How many colors you expect to need for the given clazz in this image.

        Returns:
            The color assigned to the given clazz and label, as well as whether this is the first time the given pair
            has been assigned a color.
        """
        if clazz == 'mask':
            alpha = 0.3
        else:
            alpha = 1.0
        if not hasattr(self, '_fe_color_map'):
            self._fe_color_map = {}  # ([remaining_colors], {label: assigned_color})
        if clazz not in self._fe_color_map:
            clazz_colors = get_colors(max(10, n_colors), alpha=alpha)
            clazz_assignment = {}
            self._fe_color_map[clazz] = (clazz_colors, clazz_assignment)
        clazz_colors, clazz_assignment = self._fe_color_map[clazz]
        if label in clazz_assignment:
            val = clazz_assignment[label]
            if as_numbers:
                val = [float(x) for x in val.strip('rgba(').strip(')').split(',')]
            return val, False
        if not clazz_colors:
            # The initial estimate of color requirements was insufficient
            clazz_colors.extend(get_colors(max(20, n_colors), alpha=alpha))
        clazz_assignment[label] = clazz_colors.pop(0)
        val = clazz_assignment[label]
        if as_numbers:
            val = [float(x) for x in val.strip('rgba(').strip(')').split(',')]
        return val, True

    def show(self,
             save_path: Optional[str] = None,
             verbose: bool = True,
             scale: int = 1,
             interactive: bool = True) -> None:
        """A function which will save or display plotly figures.

        Args:
            save_path: The path where the figure should be saved, or None to display the figure to the screen.
            verbose: Whether to print out the save location.
            scale: A scaling factor to apply when exporting to static images (to increase resolution).
            interactive: Whether the figure should be interactive or static. This is only applicable when
                save_path is None and when running inside a jupyter notebook. The advantage is that the file size of the
                resulting jupyter notebook can be dramatically reduced.
        """
        config = {
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'height': None,
                'width': None,
                'filename': 'figure',
                'scale': scale  # Multiply title/legend/axis/canvas sizes by this factor (high resolution save)
            }}
        if save_path is None:
            if not interactive and in_notebook():
                from IPython.display import Image, display
                display(Image(self.to_image(format='png', scale=scale)))
            else:
                super().show(config=config)
        else:
            save_path = os.path.normpath(save_path)
            root_dir = os.path.dirname(save_path)
            if root_dir == "":
                root_dir = "."
            os.makedirs(root_dir, exist_ok=True)
            save_file = os.path.join(root_dir, os.path.basename(save_path) or 'figure.html')
            config['toImageButtonOptions']['filename'] = os.path.splitext(os.path.basename(save_file))[0]
            ext = os.path.splitext(save_file)[1]
            if ext == '':
                ext = '.html'
                save_file = save_file + ext  # Use html by default
            if verbose:
                print("Saving to {}".format(save_file))
            if ext == '.html':
                self.write_html(save_file, config=config)
            else:
                self.write_image(save_file, width=1920, height=1080, scale=scale)
