import os
import tempfile
import unittest

import fastestimator as fe


class TestDirDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        a_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)
        b_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)

        a1 = open(os.path.join(a_tmpdirname.name, "fa1.txt"), "x")
        a2 = open(os.path.join(a_tmpdirname.name, "fa2.txt"), "x")

        b1 = open(os.path.join(b_tmpdirname.name, "fb1.txt"), "x")
        b2 = open(os.path.join(b_tmpdirname.name, "fb2.txt"), "x")

        dataset = fe.dataset.DirDataset(root_dir=tmpdirname)

        self.assertEqual(len(dataset), 4)
