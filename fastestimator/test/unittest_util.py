from collections import ChainMap
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

import fastestimator as fe
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.summary import System
from fastestimator.trace import Trace
from fastestimator.util.data import Data


def is_equal(obj1: Any, obj2: Any, assert_type: bool = True, assert_dtype: bool = False) -> bool:
    """Check whether input objects are equal. The object type can be nested iterable (list, tuple, set, dict) and
    with elements such as int, float, np.ndarray, tf.Tensor, tf.Varaible, torch.Tensor

    Args:
        obj1: Input object 1
        obj2: Input object 2
        assert_type: Whether to assert the same data type
        assert_dtype: Whether to assert the same dtype in case of nd.array, tf.Tensor, torch.Tensor

    Returns:
        Boolean of whether those two object are equal
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
        if assert_dtype and obj1.dtype != obj2.dtype:
            return False
        return np.array_equal(obj1, obj2)

    elif tf.is_tensor(obj1):
        if assert_dtype and obj1.dtype != obj2.dtype:
            return False
        obj1 = obj1.numpy()
        obj2 = obj2.numpy()
        return np.array_equal(obj1, obj2)

    elif isinstance(obj1, torch.Tensor):
        if assert_dtype and obj1.dtype != obj2.dtype:
            return False
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
    inp = tf.keras.layers.Input([3])
    x = tf.keras.layers.Dense(units=1, use_bias=False)(inp)
    model = tf.keras.models.Model(inputs=inp, outputs=x)
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

    """
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 1, bias=False)
        self.fc1.weight.data = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return x


class MultiLayerTorchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 2, bias=False)
        self.fc1.weight.data = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 6]], dtype=torch.float32)
        self.fc2 = torch.nn.Linear(2, 1, bias=False)
        self.fc2.weight.data = torch.tensor([[1, 2]], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MockBetaDistribution:
    def __init__(self, framework='tf'):
        self.framework = framework

    def sample(self):
        if self.framework == 'tf':
            return tf.constant(0.5)
        elif self.framework == 'torch':
            return torch.Tensor([0.5])
        else:
            raise ValueError("Unrecognized framework {}".format(self.framework))


class MockUniformDistribution:
    def __init__(self, framework='tf'):
        self.framework = framework

    def sample(self):
        if self.framework == 'tf':
            return tf.constant(0.25)
        elif self.framework == 'torch':
            return torch.Tensor([0.25])
        else:
            raise ValueError("Unrecognized framework {}".format(self.framework))


def sample_system_object():
    x_train = np.random.rand(3, 28, 28, 3)
    y_train = np.random.randint(10, size=(3, ))
    x_eval = np.random.rand(2, 28, 28, 3)
    y_eval = np.random.randint(10, size=(2, ))

    train_data = NumpyDataset({'x': x_train, 'y': y_train})
    eval_data = NumpyDataset({'x': x_eval, 'y': y_eval})
    test_data = eval_data.split(0.5)
    model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
    pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data, batch_size=1)
    network = fe.Network(ops=[ModelOp(model=model, inputs="x_out", outputs="y_pred")])
    system = System(network=network, pipeline=pipeline, traces=[], total_epochs=10, mode='train')
    return system


def sample_system_object_torch():
    x_train = np.random.rand(3, 28, 28, 3)
    y_train = np.random.randint(10, size=(3, ))
    x_eval = np.random.rand(2, 28, 28, 3)
    y_eval = np.random.randint(10, size=(2, ))

    train_data = NumpyDataset({'x': x_train, 'y': y_train})
    eval_data = NumpyDataset({'x': x_eval, 'y': y_eval})
    test_data = eval_data.split(0.5)
    model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
    pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data, batch_size=1)
    network = fe.Network(ops=[ModelOp(model=model, inputs="x_out", outputs="y_pred")])
    system = System(network=network, pipeline=pipeline, traces=[], total_epochs=10, mode='train')
    return system


def check_img_similar(img1: np.ndarray, img2: np.ndarray, ptol: int = 3, ntol: float = 0.01) -> bool:
    """Check whether img1 and img2 array are similar based on pixel to pixel comparision
    Args:
        img1: Image 1
        img2: Image 2
        ptol: Pixel value tolerance
        ntol: Number of pixel difference tolerace rate

    Returns:
        Boolean of whether the images are similar
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
        path: Image path

    Returns:
        Image numpy array
    """
    return np.asarray(Image.open(path).convert('RGB'))


def fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Convert image in plt.Figure to numpy array

    Args:
        fig: Input figure object

    Returns:
        Image array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


class TraceRun:
    """Class to simulate the trace calling protocol.

    This serve for testing purpose without using estimator class.

    Args:
        trace: Target trace to run.
        batch: Batch data from pipepline.
        prediction: Batch data from network.
    """
    def __init__(self, trace: Trace, batch: Dict[str, Any], prediction: Dict[str, Any]):
        self.trace = trace
        self.batch = batch
        self.prediction = prediction
        self.data_on_begin = None
        self.data_on_end = None
        self.data_on_epoch_begin = None
        self.data_on_epoch_end = None
        self.data_on_batch_begin = None
        self.data_on_batch_end = None


    def run_trace(self) -> None:
        self.data_on_begin = Data()
        self.trace.on_begin(self.data_on_begin)

        self.data_on_epoch_begin = Data()
        self.trace.on_epoch_begin(self.data_on_epoch_begin)

        self.data_on_batch_begin = Data(self.batch)
        self.trace.on_batch_begin(self.data_on_batch_begin)

        self.data_on_batch_end = Data(ChainMap(self.prediction, self.batch))
        self.trace.on_batch_end(self.data_on_batch_end)

        self.data_on_epoch_end = Data()
        self.trace.on_epoch_end(self.data_on_epoch_end)

        self.data_on_end = Data()
        self.trace.on_end(self.data_on_end)
