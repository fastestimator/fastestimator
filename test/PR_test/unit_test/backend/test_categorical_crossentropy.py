import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestCategoricalCrossEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_true = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cls.tf_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.torch_true = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cls.torch_pred = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])

    def test_categorical_crossentropy_average_loss_true_tf(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true).numpy()
        obj2 = 0.22839302
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_average_loss_false_tf(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true, average_loss=False).numpy()
        obj2 = np.array([0.22314353, 0.10536055, 0.35667497])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_from_logits_average_loss_true_tf(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.tf_pred,
                                                   y_true=self.tf_true,
                                                   average_loss=True,
                                                   from_logits=True).numpy()

        obj2 = 0.69182307
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_from_logits_average_loss_false_tf(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.tf_pred,
                                                   y_true=self.tf_true,
                                                   average_loss=False,
                                                   from_logits=True).numpy()

        obj2 = np.array([0.6897267, 0.6177929, 0.7679496])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_average_loss_true_torch(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true).numpy()
        obj2 = 0.22839302
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_average_loss_false_torch(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true,
                                                   average_loss=False).numpy()
        obj2 = np.array([0.22314353, 0.10536055, 0.35667497])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_from_logits__average_loss_true_torch(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.torch_pred,
                                                   y_true=self.torch_true,
                                                   average_loss=True,
                                                   from_logits=True).numpy()

        obj2 = 0.69182307
        self.assertTrue(np.allclose(obj1, obj2))

    def test_categorical_crossentropy_from_logits_average_loss_false_torch(self):
        obj1 = fe.backend.categorical_crossentropy(y_pred=self.torch_pred,
                                                   y_true=self.torch_true,
                                                   average_loss=False,
                                                   from_logits=True).numpy()

        obj2 = np.array([0.6897267, 0.6177929, 0.7679496])
        self.assertTrue(np.allclose(obj1, obj2))
