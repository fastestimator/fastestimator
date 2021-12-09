import torch
import tensorflow as tf

from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.traceability_util import traceable
from typing import Any, Dict, List, Tuple, TypeVar, Union, Iterable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class L2Regularizaton(LossOp):
    """Calculate L2 Regularization Loss.

    Args:
        inputs: String key representing input loss.
        outputs: String key under which to store the computed loss value.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".

    Raises:
        AssertionError: If `class_weights` or it's keys and values are of unacceptable data types.
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 mode: Union[None, str, Iterable[str]] = None,
                 beta: float = 0.01):
        super().__init__(inputs=inputs, outputs=outputs, mode = mode)

        self.model = model
        self.beta = beta

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Tensor:
        '''
        For pytorch: param.norm(2) is similar to `param.pow(2).sum().sqrt()`
        For tensorflow: tf.nn.l2_loss(w) is similar to `tf.reduce_sum(tf.pow(w,2)) / 2`
        '''
        loss = data
        if isinstance(self.model, torch.nn.Module):
            l2_loss = torch.tensor(0.)
            for param in self.model.parameters():
                if param.requires_grad:
                    l2_loss += (torch.sum(param.pow(2))) / 2
            total_loss = torch.add((self.beta * l2_loss), loss)

        elif isinstance(self.model, tf.keras.Model):
            l2_loss = tf.zeros(1)
            for layer in self.model.layers:
                for w in layer.trainable_variables:
                    if tf.nn.l2_loss(w) != 0.0:
                        l2_loss += tf.nn.l2_loss(w)

            total_loss = tf.add((self.beta * l2_loss)[0], loss)
        else:
            raise ValueError("Unrecognized model framework: Please make sure to pass either torch or tensorflow models")

        return total_loss
