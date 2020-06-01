import numpy as np
import tensorflow as tf

import torch
from typing import Any

def is_equal(obj1: Any, obj2: Any) -> bool:
    """Check whether input objects are equal. The object type can be nested iterable (list, tuple, set, dict) and
    with elements such as int, float, np.ndarray, tf.Tensor, tf.Varaible, torch.Tensor

    Args:
        obj1: input object 1
        obj2: input object 2

    Returns:
        boolean of whether those two object are equal
    """
    if type(obj1) != type(obj2):
        return False

    if type(obj1) in [list, set, tuple]:
        if len(obj1) != len(obj2):
            return False

        for iter1, iter2 in zip(obj1, obj2):
            if not is_equal(iter1, iter2):
                return False

        return True

    elif type(obj1) == dict:
        if len(obj1) != len(obj2):
            return False

        if obj1.keys() != obj2.keys():
            return False

        for value1, value2 in zip(obj1.values(), obj2.values()):
            if not is_equal(value1, value2):
                return False

        return True

    elif type(obj1) == np.ndarray:
        return np.array_equal(obj1, obj2)

    elif isinstance(obj1, tf.Tensor) or isinstance(obj1, tf.Variable):
        obj1 = obj1.numpy()
        obj2 = obj2.numpy()
        return np.array_equal(obj1, obj2)

    elif isinstance(obj1, torch.Tensor):
        return torch.equal(obj1, obj2)

    else:
        return obj1 == obj2


def one_layer_tf_model() -> tf.keras.Model:
    """Tensorflow Model with one dense layer without activation function.
    * Model input shape: (3,)
    * Model output: (1,)
    * dense layer weight: [1.0, 2.0, 3.0]

    How to feed_forward this model
    ```python
    model = one_layer_tf_model()
    x = tf.constant([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
    b = fe.backend.feed_forward(model, x) # [[6.0], [-2.5]]
    ```

    Returns:
        tf.keras.Model: The model
    """
    input = tf.keras.layers.Input([3])
    x = tf.keras.layers.Dense(units=1, use_bias=False)(input)
    model = tf.keras.models.Model(inputs=input, outputs=x)
    model.layers[1].set_weights([np.array([[1.0], [2.0], [3.0]])])
    return model


class OneLayerTorchModel(torch.nn.Module):
    """Torch Model with one dense layer without activation function.
    * Model input shape: (3,)
    * Model output: (1,)
    * dense layer weight: [1.0, 2.0, 3.0]

    How to feed_forward this model
    ```python
    model = OneLayerTorchModel()
    x = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
    b = fe.backend.feed_forward(model, x) # [[6.0], [-2.5]]
    ```

    Args:
        torch ([type]): The model
    """
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 1, bias=False)
        self.fc1.weight.data = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return x


