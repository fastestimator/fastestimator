import unittest

from fastestimator.schedule import EpochScheduler, RepeatScheduler, get_current_items, get_signature_epochs


class TestSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler = RepeatScheduler(['a', 'b', 'c', 'c'])
        cls.epoch_scheduler = EpochScheduler({1: 'a', 2: 'b', 30: None})
        cls.actual_current_items = ['a', 'b', 'c', 'c']
        cls.signature_epochs = [1, 2, 3, 5, 30, 31, 33]

    def test_get_current_items(self):
        current_items = get_current_items(items=[self.scheduler], run_modes="train", epoch=None)
        self.assertEqual(current_items, self.actual_current_items)

    def test_get_current_item_epoch(self):
        current_items = get_current_items(items=[self.scheduler], run_modes="train", epoch=2)
        self.assertEqual(current_items, ['b'])

    def test_get_signature_epochs(self):
        epochs = get_signature_epochs([self.scheduler, self.epoch_scheduler], total_epochs=50, mode="train")
        self.assertEqual(epochs, self.signature_epochs)
