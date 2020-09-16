import unittest

import numpy as np

from fastestimator.op.numpyop.meta import Fuse
from fastestimator.op.numpyop.univariate import Minmax


class TestFuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_shape = (28, 28, 3)
        cls.multi_input = [np.random.randint(16, size=(28, 28, 3)), np.random.randint(16, size=(28, 28, 3))]

    def test_single_input(self):
        minmax = Minmax(inputs='x', outputs='y', mode='test')
        minmax2 = Minmax(inputs=["y", "z"], outputs="w", mode='test')
        fuse = Fuse([minmax, minmax2])
        with self.subTest('Check op inputs'):
            self.assertListEqual(fuse.inputs, ['x', 'z'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(fuse.outputs, ['y', 'w'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(fuse.mode, {'test'})
        output = fuse.forward(data=self.multi_input, state={"mode": "test"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)
