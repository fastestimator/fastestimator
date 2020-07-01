import unittest

import fastestimator as fe

from fastestimator.trace import Trace


class TestTraceSortTraces(unittest.TestCase):
    """ This test has dependency on:
    * fe.trace.Trace
    * fe.util.util.to_set()
    """

    def test_estimator_sort_traces_with_available_outputs_have_dependency(self):
        trace1 = Trace(inputs="x", outputs="y")
        trace2 = Trace(inputs="y", outputs="z")
        trace3 = Trace(inputs="z", outputs="w")

        sorted_traces = fe.trace.sort_traces([trace3, trace2, trace1], available_outputs="x")
        self.assertEqual(sorted_traces, [trace1, trace2, trace3])

    def test_estimator_sort_traces_with_available_outputs_have_no_dependency(self):
        trace1 = Trace(inputs="x", outputs="y")
        trace2 = Trace(inputs="x", outputs="z")
        trace3 = Trace(inputs="x", outputs="x")

        with self.subTest("[trace1, trace2]"):
            sorted_traces = fe.trace.sort_traces([trace1, trace2], available_outputs="x")
            self.assertEqual(sorted_traces, [trace1, trace2])

        with self.subTest("[trace2, trace1]"):
            sorted_traces = fe.trace.sort_traces([trace2, trace1], available_outputs="x")
            self.assertEqual(sorted_traces, [trace2, trace1])

        with self.subTest("[trace1, trace3]"):
            sorted_traces = fe.trace.sort_traces([trace1, trace3], available_outputs="x")
            self.assertEqual(sorted_traces, [trace1, trace3])

        with self.subTest("[trace3, trace1]"):
            sorted_traces = fe.trace.sort_traces([trace3, trace1], available_outputs="x")
            self.assertEqual(sorted_traces, [trace3, trace1])

    def test_estimator_sort_traces_with_available_outputs_no_input_match(self):
        trace1 = Trace(inputs="y", outputs="z")
        with self.assertRaises(AssertionError):
            sorted_traces = fe.trace.sort_traces([trace1], available_outputs="x")

    def test_estimator_sort_traces_without_available_outputs_have_dependency(self):
        trace1 = Trace(inputs="x", outputs="y")
        trace2 = Trace(inputs="y", outputs="z")
        trace3 = Trace(inputs="z", outputs="w")

        sorted_traces = fe.trace.sort_traces([trace3, trace2, trace1])
        self.assertEqual(sorted_traces, [trace1, trace2, trace3])

    def test_estimator_sort_traces_without_available_outputs_have_no_dependency(self):
        trace1 = Trace(inputs="x", outputs="y")
        trace2 = Trace(inputs="a", outputs="b")

        with self.subTest("[trace1, trace2]"):
            sorted_traces = fe.trace.sort_traces([trace1, trace2])
            self.assertEqual(sorted_traces, [trace1, trace2])

        with self.subTest("[trace2, trace1]"):
            sorted_traces = fe.trace.sort_traces([trace2, trace1])
            self.assertEqual(sorted_traces, [trace2, trace1])

    def test_estimator_sort_traces_cycle_dependency(self):
        trace1 = Trace(inputs="x", outputs="y")
        trace2 = Trace(inputs="y", outputs="z")
        trace3 = Trace(inputs="z", outputs="x")
        trace4 = Trace(inputs="x", outputs="a")

        with self.subTest("available_outputs='x'"):
            sorted_traces = fe.trace.sort_traces([trace1, trace2, trace3, trace4], available_outputs="y")
            self.assertEqual(sorted_traces, [trace2, trace3, trace4, trace1])

        with self.subTest("available_output=None"):
            with self.assertRaises(AssertionError):
                sorted_traces = fe.trace.sort_traces([trace1, trace2, trace3, trace4])
