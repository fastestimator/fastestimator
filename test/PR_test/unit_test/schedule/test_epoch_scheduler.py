import unittest

from fastestimator.schedule import EpochScheduler


class TestEpochScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_data = {1: "a", 3: "b", 4: None, 100: "c"}
        cls.values = ['a', 'b', None, 'c']
        cls.scheduler = EpochScheduler(cls.input_data)

    def test_get_current_value(self):
        self.assertEqual(self.scheduler.get_current_value(2), 'a')

    def test_get_all_values(self):
        self.assertEqual(self.scheduler.get_all_values(), self.values)

    def test_get_last_key(self):
        self.assertEqual(self.scheduler._get_last_key(3), 3)
