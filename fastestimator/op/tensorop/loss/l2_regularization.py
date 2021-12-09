import torch
import tensorflow as tf

from fastestimator.op.tensorop import TensorOp
from typing import Any, Dict, List, Tuple, TypeVar, Union

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)

class L2Regularizaton(TensorOp):
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 beta: float = 0.01):
        super().__init__(inputs=inputs, outputs=outputs)

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
                    if tf.reduce_sum(tf.pow(w, 2)) != 0.0:
                        l2_loss += tf.reduce_sum(tf.pow(w, 2)) / 2

            total_loss = tf.add((self.beta * l2_loss)[0], loss)

        return total_loss
