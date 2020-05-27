import unittest

class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj = [1,2,3]
        print("run setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("run tearDownClass")

    def setUp(self):
        print("run setUp")

    def tearDown(self):
        print("run tearDown")

    def test_dummy(self):
        print("run test")
        print(self.obj)
        self.obj.append(4)
        self.assertEqual(1,1)

    def test_dummy2(self):
        print("run test2")
        print(self.obj)
        self.assertEqual(1, 1)
