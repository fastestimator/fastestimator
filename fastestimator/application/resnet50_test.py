"""Unit tests for resnet50.py."""
import numpy as np
import tensorflow as tf

from fastestimator.application.resnet50 import ResNet50


def test_resnet50_pass_forward():
    """Feed in random numpy array and check if the output dimensions are expected."""
    tf.keras.backend.clear_session()
    model = ResNet50('x', 'y', (256, 256 ,1), classes=2)
    pred = model.predict(np.random.rand(1, 256, 256, 1))
    assert len(pred[0]) == 2
    tf.keras.backend.clear_session()
