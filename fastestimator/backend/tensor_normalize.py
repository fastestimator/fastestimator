from typing import TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend_config import epsilon
import torch

from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_std import reduce_std
from fastestimator.backend import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)
from typing import Sequence, TypeVar, Union

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def normalize(tensor: Tensor, mean, std, epsilon: float = 1e-7) -> Tensor:
    """Compute the std value along a given `axis` of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(n)  # 2.2913
    b = fe.backend.reduce_std(n, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(n, axis=1)  # [[1., 1.], [1., 1.]]
    b = fe.backend.reduce_std(n, axis=[0,2])  # [2.06155281, 2.06155281]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(t)  # 2.2913
    b = fe.backend.reduce_std(t, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(t, axis=1)  # [[2, 3], [3, 7]]
    b = fe.backend.reduce_std(t, axis=[0,2])  # [2.06155281, 2.06155281]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(p)  # 2.2913
    b = fe.backend.reduce_std(p, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(p, axis=1)  # [[1., 1.], [1., 1.]]
    b = fe.backend.reduce_std(p, axis=[0,2])  # [2.06155281, 2.06155281]
    ```

    Args:
        tensor: The input value.
        axis: Which axis or collection of axes to compute the std along.
        keepdims: Whether to preserve the number of dimensions during the reduction.

    Returns:
        The std values of `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    mean = get_mean(tensor, mean)
    std = get_std(tensor, std)

    if tf.is_tensor(tensor) or isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
        tensor = (tensor - mean) / std
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.permute((0, 3, 1, 2))  # channel first
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return tensor


def get_mean(tensor: Tensor, mean):
    if mean == None:
        return reduce_mean(tensor)

    if tf.is_tensor(tensor):
        mean = tf.convert_to_tensor(mean)
    elif isinstance(tensor, torch.Tensor):
        mean = to_tensor(mean, "torch").type(torch.float32)
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
    
    return mean


def get_std(tensor: Tensor, std):

    if std == None:
        return reduce_std(tensor)

    if tf.is_tensor(tensor):
        std = tf.convert_to_tensor(std)
    elif isinstance(tensor, torch.Tensor):
        std = to_tensor(std, "torch").type(torch.float32)
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
    
    return std


