import unittest

from fastestimator.schedule import RepeatScheduler


class TestRepeatScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_data = [1, 2, 2]
        cls.scheduler = RepeatScheduler(cls.input_data)

    def test_repeat_scheduler_current_value(self):
        self.assertEqual(self.scheduler.get_current_value(epoch=2), 2)

    def test_repeat_scheduler_all_values(self):
        self.assertEqual(self.scheduler.get_all_values(), self.input_data)
