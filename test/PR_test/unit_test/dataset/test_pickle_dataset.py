import unittest

import fastestimator as fe


class TestPickleDataset(unittest.TestCase):
    def test_dataset(self):
        test_data = fe.dataset.PickleDataset("./dummy.pkl")

        self.assertEqual(len(test_data), 5)
