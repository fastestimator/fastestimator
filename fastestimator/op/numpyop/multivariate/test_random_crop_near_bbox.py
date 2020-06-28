import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import RandomCropNearBBox


class TestRandomCropNearBBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.rand(28, 28, 3), np.array([12, 12, 19, 19])]
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3), np.array([12, 12, 19, 19])]

    def test_input(self):
        randomcrop_bbox = RandomCropNearBBox(image_in='x', cropping_bbox_in="x_bbox")
        output = randomcrop_bbox.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)

    def test_input_image_and_mask(self):
        randomcrop_bbox = RandomCropNearBBox(image_in='x', mask_in='x_mask', cropping_bbox_in="x_bbox")
        output = randomcrop_bbox.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[0].shape, output[1].shape)
