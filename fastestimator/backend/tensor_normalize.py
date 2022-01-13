from typing import Tuple, TypeVar

import torch
import numpy as np
import tensorflow as tf


from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_std import reduce_std
from fastestimator.backend.to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def normalize(tensor: Tensor, mean, std, epsilon: float = 1e-7) -> Tensor:
    """Compute the normalized value of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, None, None)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, 4.5, 2.29128784747792)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (4., 5.), (2.23606798, 2.23606798))  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, None, None)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, 4.5, 2.29128784747792)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (4., 5.), (2.23606798, 2.23606798))  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, None, None)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, 4.5, 2.29128784747792)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (4., 5.), (2.23606798, 2.23606798))  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    Args:
        tensor: The input value.
        mean: The mean which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))
        std: The standard deviation which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))

    Returns:
        The normalized values of `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    channel_level = False
    if isinstance(mean, Tuple) or isinstance(std, Tuple):
        channel_level = True

    mean = get_mean(tensor, mean, channel_level)
    std = get_std(tensor, std, channel_level)
    epsilon = get_epsilon(tensor, epsilon)

    if tf.is_tensor(tensor) or isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
        tensor = (tensor - mean) / (std + epsilon)

        # for testing purpose we are not changing the channels
        #if isinstance(tensor, torch.Tensor):
        #    tensor = tensor.permute((0, 1, 2, 3))  # channel first
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return tensor


def get_mean(tensor: Tensor, mean, channel_level):
    if mean == None:
        if channel_level:
            return reduce_mean(tensor, axis=list(range(0, len(tensor.shape)-1)))
        else:
            return reduce_mean(tensor)

    if tf.is_tensor(tensor):
        mean = tf.cast(tf.convert_to_tensor(mean), tf.float32)
    elif isinstance(tensor, torch.Tensor):
        mean = to_tensor(mean, "torch").type(torch.float32)
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return mean


def get_epsilon(tensor: Tensor, epsilon):

    if tf.is_tensor(tensor):
        return tf.cast(tf.convert_to_tensor(epsilon), tf.float32)
    elif isinstance(tensor, torch.Tensor):
        return to_tensor(epsilon, "torch").type(torch.float32)
    elif isinstance(tensor, np.ndarray):
        return epsilon
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def get_std(tensor: Tensor, std, channel_level):

    if std == None:
        if channel_level:
            return reduce_std(tensor, axis=list(range(0, len(tensor.shape)-1)))
        else:
            return reduce_std(tensor)

    if tf.is_tensor(tensor):
        std = tf.cast(tf.convert_to_tensor(std), tf.float32)
    elif isinstance(tensor, torch.Tensor):
        std = to_tensor(std, "torch").type(torch.float32)
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
    
    return std


