import unittest

import numpy as np

import fastestimator as fe
from fastestimator.dataset import GeneratorDataset


def inputs():
    while True:
        yield {"x": np.random.rand(16), "y": np.random.randint(16)}


def inputs2():
    while True:
        yield {"x1": np.random.rand(16), "y1": np.random.randint(16)}


def inputs3():
    while True:
        yield (np.random.rand(16), np.random.randint(16))


class TestCombinedDataset(unittest.TestCase):
    def test_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        combined_ds = fe.dataset.CombinedDataset(datasets=[ds1, ds2])

        self.assertEqual(len(combined_ds), 20)
        sample = combined_ds[12]
        self.assertEqual(type(sample), dict)
        self.assertEqual(list(sample.keys()), ["x", "y"])

    def test_error_dataset(self):
        with self.assertRaises(AssertionError) as err_msg:
            fe.dataset.CombinedDataset([])
        self.assertEqual("datasets should not be an empty iterable", str(err_msg.exception))

    def test_invalid_index_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        combined_ds = fe.dataset.CombinedDataset(datasets=[ds1, ds2])

        with self.assertRaises(IndexError) as err_msg:
            combined_ds[20]

        self.assertEqual(
            "list index out of range",
            str(err_msg.exception),
        )

    def test_invalid_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        with self.assertRaises(AssertionError) as err_msg:
            combined_ds = fe.dataset.CombinedDataset(datasets=[ds1, "ds2", "ds2"])
        self.assertEqual(
            "Each dataset should be a type of PyTorch Dataset and should return a dictionary.",
            str(err_msg.exception),
        )

    def test_invalid_keys(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs2(), samples_per_epoch=5)
        with self.assertRaises(KeyError) as err_msg:
            combined_ds = fe.dataset.CombinedDataset(datasets=[ds1, ds2])
        self.assertEqual("All datasets should have the same keys.", err_msg.exception.args[0])

    def test_invalid_return_type_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs3(), samples_per_epoch=5)
        with self.assertRaises(AssertionError) as err_msg:
            combined_ds = fe.dataset.CombinedDataset(datasets=[ds1, ds2])
        self.assertEqual(
            "Each dataset should be a type of PyTorch Dataset and should return a dictionary.",
            str(err_msg.exception),
        )
