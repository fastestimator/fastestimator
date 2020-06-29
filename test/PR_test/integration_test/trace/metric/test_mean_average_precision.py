import unittest

import numpy as np

from fastestimator.trace.metric import MeanAveragePrecision
from fastestimator.util import Data


class TestMeanAveragePrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.random.rand(1, 5, 5)
        x_pred = np.random.rand(1, 5, 7)
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.map = MeanAveragePrecision(true_key='x', pred_key='x_pred', num_classes=3)
        cls.iou_element_shape = (5, 5)

    def test_on_epoch_begin(self):
        self.map.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of image ids'):
            self.assertEqual(self.map.image_ids, [])
        with self.subTest('Check initial value of eval images'):
            self.assertEqual(self.map.evalimgs, {})
        with self.subTest('Check initial value of eval'):
            self.assertEqual(self.map.eval, {})
        with self.subTest('Check initial value of ids in epoch'):
            self.assertEqual(self.map.ids_in_epoch, 0)

    def test_reshape_gt(self):
        x = np.random.rand(1, 5, 5)
        output = self.map._reshape_gt(x)
        self.assertEqual(output.shape, (5, 6))

    def test_reshape_pred(self):
        x = np.random.rand(2, 5, 7)
        output = self.map._reshape_pred(x)
        self.assertEqual(output.shape, (10, 7))

    def test_on_batch_end(self):
        self.map.on_batch_end(data=self.data)
        with self.subTest('Check IOU element shape'):
            self.assertEqual(self.map.ious[(1, 0)].shape, (5, 5))

    def test_on_epoch_end(self):
        self.map.ious = {(1, 0): np.random.uniform(low=0, high=1, size=(5, 5)), (1, 1): [], (1, 2): []}
        self.evalimgs = {
            (0, 1): {
                'image_id': 1,
                'category_id': 0,
                'gtIds': [1, 1, 1, 1, 1],
                'dtMatches': np.random.randint(2, size=(10, 5)),
                'gtMatches': np.random.randint(2, size=(10, 5)),
                'dtScores': np.random.uniform(low=0, high=1, size=(5, )),
                'num_gt': 5
            }, (1, 1): None, (2, 1): None
        }
        self.map.on_epoch_end(data=self.data)
        with self.subTest('Check if mAP exists'):
            self.assertIn('mAP', self.data)
        with self.subTest('Check if AP50 exists'):
            self.assertIn('AP50', self.data)
        with self.subTest('Check if AP75 exists'):
            self.assertIn('AP75', self.data)
        with self.subTest('Check the value of mAP'):
            self.assertEqual(self.data['mAP'], -1)
        with self.subTest('Check the value of AP50'):
            self.assertEqual(self.data['AP50'], -1)
        with self.subTest('Check the value of AP75'):
            self.assertEqual(self.data['AP75'], -1)
