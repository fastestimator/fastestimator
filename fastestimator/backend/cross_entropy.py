import tensorflow as tf
import torch


def cross_entropy(y_pred, y_true, apply_softmax=False):
    """calculate cross entropy for tensor inputs

    Args:
        y_pred (tf.Tensor or torch.Tensor): prediction score for each class, in [Batch, C]
        y_true (tf.Tensor or torch.Tensor): ground truth class label index, in [Batch]
        apply_softmax (bool, optional): whether to apply softmax to y_pred. Defaults to False.

    Returns:
        [tf.Tensor or torch.Tensor]: categorical cross entropy
    """
    assert type(y_pred) == type(y_true), "y_pred and y_true must be same tensor type"
    assert isinstance(y_pred, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_pred"
    assert isinstance(y_true, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_true"
    if isinstance(y_pred, tf.Tensor):
        ce = tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=apply_softmax)
    else:
        if apply_softmax:
            ce = torch.nn.CrossEntropyLoss(reduction="none")(y_pred, y_true)
        else:
            ce = torch.nn.NLLLoss(reduction="none")(torch.log(y_pred), y_true)
    return ce
