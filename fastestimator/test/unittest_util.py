from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from PIL import Image


def is_equal(obj1: Any, obj2: Any, assert_type: bool = True) -> bool:
    """Check whether input objects are equal. The object type can be nested iterable (list, tuple, set, dict) and
    with elements such as int, float, np.ndarray, tf.Tensor, tf.Varaible, torch.Tensor

    Args:
        obj1: input object 1
        obj2: input object 2
        assert_dtype: whether to assert the same data type

    Returns:
        boolean of whether those two object are equal
    """
    if assert_type and type(obj1) != type(obj2):
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

    elif tf.is_tensor(obj1)::
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


def check_img_similar(img1: np.ndarray, img2: np.ndarray, ptol: int = 3, ntol: float = 0.01) -> bool:
    """ check whether img1 and img2 array are similar based on pixel to pixel comparision
    Args:
        img1: image 1
        img2: image 2
        ptol: pixel value tolerance
        ntol: number of pixel difference tolerace rate

    Returns:
        boolean of whether the images are similar
    """
    if img1.shape == img2.shape:
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        n_pixel_diff = diff[diff > ptol].size
        if n_pixel_diff < img1.size * ntol:
            return True
        else:
            return False
    return False


def img_to_rgb_array(path: str) -> np.ndarray:
    """Read png file to numpy array (RGB)

    Args:
        path: image path

    Returns:
        image nump yarray
    """
    return np.asarray(Image.open(path).convert('RGB'))


def fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """convert image in plt.Figure to numpy array

    Args:
        fig: input figure object

    Returns:
        image array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
