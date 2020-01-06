import tensorflow as tf
import torch


def reduce_loss(loss):
    if isinstance(loss, tf.Tensor):
        assert len(loss.shape) < 2, "loss must be one-dimentional or scalar"
        if len(loss.shape) == 1:
            loss = tf.reduce_mean(loss)
    elif isinstance(loss, torch.Tensor):
        assert loss.ndim < 2, "loss must be one-dimentional or scalar"
        if loss.ndim == 1:
            loss = torch.mean(loss)
    else:
        raise ValueError("loss must be either tf.Tensor or torch.Tensor")
    return loss
