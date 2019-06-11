"""Unit tests for inceptionv3.py."""
import numpy as np
import tensorflow as tf

from fastestimator.application.inceptionv3 import InceptionV3


def test_inceptionv3_pass_forward():
    """Feed in random numpy array and check if the output dimensions are expected."""
    tf.keras.backend.clear_session()
    model = InceptionV3('x', 'y', (256, 256 ,1), classes=2)
    pred = model.predict(np.random.rand(1, 256, 256, 1))
    assert len(pred[0]) == 2