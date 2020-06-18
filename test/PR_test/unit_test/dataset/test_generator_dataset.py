import unittest

import numpy as np

from fastestimator.dataset import GeneratorDataset


def inputs():
    while True:
        yield {'x': np.random.rand(4), 'y': np.random.randint(2)}


class TestGeneratorDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = GeneratorDataset(generator=inputs(), samples_per_epoch=10)

        self.assertEqual(len(dataset), 10)


