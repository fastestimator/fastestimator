import tensorflow as tf
import torch


def to_number(data):
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    elif isinstance(data, torch.Tensor):
        data = data.data.numpy()
    return data
