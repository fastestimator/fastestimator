import os
import tempfile
import unittest

import fastestimator as fe


class TestLabeledDirDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        a_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)
        b_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)

        a1 = open(os.path.join(a_tmpdirname.name, "a1.txt"), "x")
        a2 = open(os.path.join(a_tmpdirname.name, "a2.txt"), "x")

        b1 = open(os.path.join(b_tmpdirname.name, "b1.txt"), "x")
        b2 = open(os.path.join(b_tmpdirname.name, "b2.txt"), "x")

        dataset = fe.dataset.LabeledDirDataset(root_dir=tmpdirname)

        self.assertEqual(len(dataset), 4)
