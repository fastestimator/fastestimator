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
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ContextDecorator
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
import torch.backends.mps
from cpuinfo import get_cpu_info
from pyfiglet import Figlet
from tensorflow.python.ops.logging_ops import print_v2

from fastestimator.util.base_util import warn

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
    torch.bool: bool,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.float16: np.float16,
    tf.uint8: np.uint8,
    tf.int8: np.int8,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.bool: bool,
    np.dtype('float32'): np.float32,
    np.dtype('float64'): np.float64,
    np.dtype('float16'): np.float16,
    np.dtype('uint8'): np.uint8,
    np.dtype('int8'): np.int8,
    np.dtype('int16'): np.int16,
    np.dtype('int32'): np.int32,
    np.dtype('int64'): np.int64,
    np.dtype('bool'): bool,
}

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
T = TypeVar('T')


class Suppressor(object):
    """A class which can be used to silence output of function calls.

    This class is intentionally not @traceable.

    Args:
        allow_pyprint: Whether to allow python printing to occur still within this scope (and therefore only silence
            printing from non-python sources like c).
        show_if_exception: Whether to retroactively show print messages if an exception is raised from within the
            suppressor scope.

    ```python
    x = lambda: print("hello")
    x()  # "hello"
    with fe.util.Suppressor():
        x()  #
    x()  # "hello"
    ```
    """
    # Only create one file to save on disk IO
    stash_fd, stash_name = tempfile.mkstemp()
    os.close(stash_fd)
    tf_print_fd, tf_print_name = tempfile.mkstemp()
    os.close(tf_print_fd)
    tf_print_name_f = 'file://' + tf_print_name

    def __init__(self, allow_pyprint: bool = False, show_if_exception: bool = False):
        self.allow_pyprint = allow_pyprint
        self.show_if_exception = show_if_exception

    def __enter__(self) -> None:
        # This is not necessary to block printing, but lets the system know what's happening
        self.py_reals = [sys.stdout, sys.stderr]
        sys.stdout = sys.stderr = self
        # This part does the heavy lifting
        if self.show_if_exception:
            self.fake = os.open(self.stash_name, os.O_RDWR)
        else:
            self.fake = os.open(os.devnull, os.O_RDWR)
        self.reals = [os.dup(1), os.dup(2)]  # [stdout, stderr]
        os.dup2(self.fake, 1)
        os.dup2(self.fake, 2)
        # This avoids "OSError: [WinError 6] The handle is invalid" while logging tensorflow information in windows
        for handler in tf.get_logger().handlers:
            handler.setStream(sys.stderr)
        if self.allow_pyprint:
            tf.print = _custom_tf_print

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        # If there was an error, display any print messages
        if exc[0] is not None and self.show_if_exception and not isinstance(exc[1],
                                                                            (StopIteration, StopAsyncIteration)):
            for line in open(self.stash_name):
                os.write(self.reals[0], line.encode('utf-8'))
        # Set the print pointers back
        os.dup2(self.reals[0], 1)
        os.dup2(self.reals[1], 2)
        # Set the python pointers back too
        sys.stdout, sys.stderr = self.py_reals[0], self.py_reals[1]

        for handler in tf.get_logger().handlers:
            handler.setStream(sys.stderr)

        # Clean up the descriptors
        for fd in self.reals:
            os.close(fd)
        os.close(self.fake)
        if self.show_if_exception:
            # Clear the file
            open(self.stash_name, 'w').close()
        if self.allow_pyprint:
            tf.print = print_v2
            with open(self.tf_print_name, 'r') as f:
                for line in f:
                    print(line, end='')  # Endings already included from tf.print
            open(self.tf_print_name, 'w').close()

    def write(self, dummy: str) -> None:
        """A function which is invoked during print calls.

        Args:
            dummy: The string which wanted to be printed.
        """
        if self.allow_pyprint:
            os.write(self.reals[0], dummy.encode('utf-8'))
        elif self.show_if_exception:
            os.write(self.fake, dummy.encode('utf-8'))

    def flush(self) -> None:
        """A function to empty the current print buffer. No-op in this case.
        """

    @staticmethod
    @atexit.register
    def teardown() -> None:
        """Clean up the stash files when the program exists
        """
        try:
            os.remove(Suppressor.stash_name)
            os.remove(Suppressor.tf_print_name)
        except FileNotFoundError:
            pass


def _custom_tf_print(*args, **kwargs):
    kwargs['output_stream'] = Suppressor.tf_print_name_f
    print_v2(*args, **kwargs)


def is_valid_file(file_path: str) -> bool:
    """Validate whether file is valid or not.

    Args:
        file_path: location of the input file.

    Returns:
        Whether the file is valid.
    """
    if not os.path.exists(file_path):
        return False
    suffix = Path(file_path).suffix
    try:
        if suffix == '.zip':
            import zipfile
            zip_file = zipfile.ZipFile(file_path)
            _ = zip_file.namelist()
        elif suffix == '.gz':
            if file_path.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(file_path) as img_tar:
                    _ = img_tar.getmembers()
            else:
                import gzip
                f = gzip.open(file_path, 'rb')
                _ = f.read()
        return True
    except Exception as e:
        print(e)
        return False


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
    """Pad `data` by appending `pad_value`s along it's dimensions until the `target_shape` is reached. All entries of
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


def move_tensors_to_device(data: T, device: Union[str, torch.device]) -> T:
    """Move torch tensor (collections) between gpu and cpu recursively.

    Args:
        data: The input data to be moved.
        device: The target device.

    Returns:
        Output data.
    """
    if isinstance(data, dict):
        return {key: move_tensors_to_device(value, device) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(val, device) for val in data]
    elif isinstance(data, tuple):
        return tuple([move_tensors_to_device(val, device) for val in data])
    elif isinstance(data, set):
        return set([move_tensors_to_device(val, device) for val in data])
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def detach_tensors(data: T) -> T:
    """Detach tensor (collections) from current graph recursively.

    Args:
        data: The data to be detached.

    Returns:
        Output data.
    """
    if isinstance(data, dict):
        return {key: detach_tensors(value) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [detach_tensors(val) for val in data]
    elif isinstance(data, tuple):
        return tuple([detach_tensors(val) for val in data])
    elif isinstance(data, set):
        return set([detach_tensors(val) for val in data])
    elif isinstance(data, torch.Tensor):
        return data.detach()
    return data


@lru_cache()
def get_device() -> torch.device:
    """Get the torch device for the current hardware.

    Returns:
        The torch device most appropriate for the current hardware.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


@lru_cache()
def get_num_gpus() -> int:
    """Get the number of GPUs available.

    Returns:
        The number of GPUs available.
    """
    if torch.backends.mps.is_available():
        return 1
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0


@lru_cache()
def get_gpu_info() -> List[str]:
    """Get summaries of all of the GPUs accessible on this machine.

    Returns:
        A formatted summary of the GPUs available on the machine (one list entry per GPU).
    """
    if shutil.which('nvidia-smi') is not None:
        nvidia_command = ['nvidia-smi', '--query-gpu=gpu_name,memory.total,driver_version', '--format=csv']
        output = subprocess.check_output(nvidia_command)
        output = output.decode('utf-8')
        lines = output.strip().split(os.linesep)[1:]
        names = []
        for line in lines:
            name, mem, driver = line.strip().split(',')
            names.append(f"{name.strip()} ({mem.strip()}, Driver={driver.strip()})")
        return names
    elif torch.backends.mps.is_available():
        output = subprocess.check_output(["ioreg", "-l"])
        output = output.decode('utf-8')
        core_count = re.search(r'"gpu-core-count"[ ]*=[ ]*(\d*)', output)
        core_count = "???" if not core_count else core_count.group(1)
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        output = output.decode('utf-8')
        gpu_mem = f"{float(output)*1e-9:0.2f} GB"  # Convert from bytes to GB
        return [f"{get_cpu_info()['brand_raw']} ({gpu_mem}, {core_count} Cores)"]  # On mac the CPU name is the GPU name
    return []


@lru_cache()
def get_num_devices() -> int:
    """Determine the number of available GPUs.

    Returns:
        The number of available GPUs, or 1 if none are found.
    """
    return max(torch.cuda.device_count(), 1)


@lru_cache()
def cpu_count(limit: Optional[int] = None) -> int:
    """Determine the number of available CPUs (correcting for docker container limits).

    Args:
        limit: If provided, the TF and Torch backends will be told to use `limit` number of threads, or the available
            number of cpus if the latter is lower (`limit` cannot raise the number of threads). A limit can only be
            enforced once per python session, before starting anything like pipeline which requires multiprocessing.

    Returns:
        The nuber of available CPUs (correcting for docker container limits), or the user provided `limit`.

    Raises:
        ValueError: If a `limit` is provided which doesn't match previously enforced limits.
    """
    existing_limit = os.environ.get('FE_NUM_THREADS_', None)  # This variable is used internally to indicate whether cpu
    # limits have already been enforced in this python session
    if existing_limit:
        try:
            existing_limit = int(existing_limit)
        except ValueError as err:
            print("FastEstimator-Error: FE_NUM_THREADS_ is an internal variable. Use FE_NUM_THREADS (no underscore)")
            raise err
        if limit and limit != existing_limit:
            raise ValueError(f"Tried to enforce a cpu limit of {limit}, but {existing_limit} was already set.")
        return existing_limit
    # Check if user provided an environment variable limit on the number of threads
    env_limit = os.environ.get('FE_NUM_THREADS', None)  # User might set this one in a bash script
    if env_limit:
        try:
            env_limit = int(env_limit)
        except ValueError as err:
            warn(f"FE_NUM_THREADS variable must be an integer, but was set to: {env_limit}")
            raise err
    try:
        # In docker containers which have --cpuset-cpus, the limit won't be reflected by normal os.cpu_count() call
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        # Running on Mac or Windows where the above method isn't available, so use the regular way
        cores = os.cpu_count()
    cores = min(cores, limit or cores, env_limit or cores)
    if cores < 1:
        raise ValueError(f"At least 1 core is required for training, but found {cores}")
    os.environ['FE_NUM_THREADS_'] = f"{cores}"  # Remember the value so we don't try to re-set the frameworks later
    os.environ['OMP_NUM_THREADS'] = f"{cores}"
    os.environ['MKL_NUM_THREADS'] = f"{cores}"
    os.environ['TF_NUM_INTEROP_THREADS'] = f"{cores}"
    os.environ['TF_NUM_INTRAOP_THREADS'] = f"{cores}"
    torch.set_num_threads(cores)
    torch.set_num_interop_threads(cores)
    return cores


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


def to_number(data: Union[tf.Tensor, torch.Tensor, np.ndarray, int, float, str]) -> np.ndarray:
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
