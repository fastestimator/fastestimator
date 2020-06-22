import os
import tempfile
import unittest

import pandas as pd

import fastestimator as fe


class TestCSVDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        data = {'x': ['a1.txt', 'a2.txt', 'b1.txt', 'b2.txt'], 'y': [0, 0, 1, 1]}
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(tmpdirname, 'data.csv'), index=False)

        dataset = fe.dataset.CSVDataset(file_path=os.path.join(tmpdirname, 'data.csv'))

        self.assertEqual(len(dataset), 4)
