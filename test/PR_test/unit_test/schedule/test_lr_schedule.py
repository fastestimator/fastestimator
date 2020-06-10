import math
import unittest

from fastestimator.schedule.lr_shedule import cosine_decay


class TestLRSchedule(unittest.TestCase):
    def test_cosine_decay(self):
        learning_rate = cosine_decay(time=5, cycle_length=10, init_lr=0.01, min_lr=0.0001)
        self.assertTrue(math.isclose(learning_rate, 0.006579, rel_tol=1e-3))

    def test_cosine_decay_cycle(self):
        learning_rate = cosine_decay(time=11, cycle_length=10, init_lr=0.01, min_lr=0.0001)
        self.assertEqual(learning_rate, 0.01)


if __name__ == "__main__":
    unittest.main()
