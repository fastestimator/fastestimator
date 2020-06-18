import unittest

import fastestimator as fe


class TestSummary(unittest.TestCase):
    def test_merge(self):
        summary1 = fe.summary.Summary(name='test1')
        summary1.history['e1']['metrics'] = {'acc': 0.9}

        summary2 = fe.summary.Summary(name='test2')
        summary2.history['e1']['metrics'] = {'acc': 0.8}

        summary1.merge(summary2)

        self.assertEqual(summary1.history['e1']['metrics']['acc'], 0.8)
