"""Unit tests for lenet.py."""
import numpy as np
import tensorflow as tf

from fastestimator.application.lenet import LeNet


def test_lenet_pass_forward_with_random_weights():
    """Feed in random numpy array and check if the network outputs are expected."""
    tf.keras.backend.clear_session()
    np.random.seed(133)
    model = LeNet('x', 'y', random_seed=66)
    pred = model.predict(np.random.rand(1, 28, 28, 1))
    expected_pred = [[0.07537358, 0.10833206, 0.08203267, 0.07823766, 0.10359581,
                      0.13456498, 0.10024161, 0.09882604, 0.11215878, 0.10663676]]

    np.testing.assert_allclose(pred, expected_pred, rtol=1e-4)
