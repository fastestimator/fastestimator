import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestBinarayCrossEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_true = tf.constant([[1], [0], [1], [0]])
        cls.tf_pred = tf.constant([[0.9], [0.3], [0.8], [0.1]])
        cls.torch_true = torch.tensor([[1], [0], [1], [0]])
        cls.torch_pred = torch.tensor([[0.9], [0.3], [0.8], [0.1]])

    def test_binaray_crossentropy_average_loss_true_tf(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true).numpy()
        obj2 = 0.19763474
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_loss_false_tf(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true, average_loss=False).numpy()
        obj2 = np.array([0.10536041, 0.3566748, 0.22314338, 0.10536041])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_from_logit_average_loss_true_tf(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.tf_pred,
                                              y_true=self.tf_true,
                                              from_logits=True,
                                              average_loss=True).numpy()
        obj2 = 0.57775164
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_from_logit_average_loss_false_tf(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.tf_pred,
                                              y_true=self.tf_true,
                                              from_logits=True,
                                              average_loss=False).numpy()
        obj2 = np.array([0.34115386, 0.8543553, 0.37110066, 0.7443967])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_loss_true_torch(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true).numpy()
        obj2 = 0.19763474
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_loss_false_torch(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true,
                                              average_loss=False).numpy()
        obj2 = np.array([0.10536041, 0.3566748, 0.22314338, 0.10536041])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_from_logit_average_loss_true_torch(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.torch_pred,
                                              y_true=self.torch_true,
                                              from_logits=True,
                                              average_loss=True).numpy()
        obj2 = 0.57775164
        self.assertTrue(np.allclose(obj1, obj2))

    def test_binaray_crossentropy_average_from_logit_average_loss_false_torch(self):
        obj1 = fe.backend.binary_crossentropy(y_pred=self.torch_pred,
                                              y_true=self.torch_true,
                                              from_logits=True,
                                              average_loss=False).numpy()
        obj2 = np.array([0.34115386, 0.8543553, 0.37110066, 0.7443967])
        self.assertTrue(np.allclose(obj1, obj2))
