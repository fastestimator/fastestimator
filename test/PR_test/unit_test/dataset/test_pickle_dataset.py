import os
import unittest

import fastestimator as fe


class TestPickleDataset(unittest.TestCase):
    def test_dataset(self):
        test_data = fe.dataset.PickleDataset(os.path.abspath(os.path.join(__file__, "..", "resources", "dummy.pkl")))

        self.assertEqual(len(test_data), 5)
