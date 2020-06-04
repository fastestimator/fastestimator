import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
import unittest
import time
from unittest.mock import patch
from io import StringIO
import pdb
from fastestimator.test.unittest_util import is_equal
class TestParseStringToPython(unittest.TestCase):
    def test_parse_string_to_python_positive_int(self):
        x = fe.util.parse_string_to_python("5")
        self.assertEqual(x, 5)

    def test_parse_string_to_python_negative_float(self):
        x = fe.util.parse_string_to_python("-1.5")
        self.assertEqual(x, -1.5)

    def test_parse_string_to_python_list(self):
        x = fe.util.parse_string_to_python("[5, 4, 0.3]")
        self.assertTrue(x, [5, 4, 0.3])

    def test_parse_string_to_python_list_over_mixed_entries(self):
        x = fe.util.parse_string_to_python("[5, [1,2,3]]")
        self.assertTrue(x, [5, [1,2,3]])

    def test_parse_string_to_python_dict(self):
        x = fe.util.parse_string_to_python("{'a': 5, 'b': 7}")
        self.assertEqual(x, {'a': 5, 'b': 7})

    def test_parse_string_to_python_dict_over_mixed_entries(self):
        x = fe.util.parse_string_to_python("{'a':5, 'b':[1,2,3]}")
        self.assertEqual(x, {'a': 5, 'b':[1,2,3]})

class TestToList(unittest.TestCase):
    def test_to_list_input_none(self):
        x = fe.util.to_list(None)
        self.assertEqual(x, [])

    def test_to_list_input_float(self):
        x = fe.util.to_list(0.5)
        self.assertEqual(x, [0.5])

    def test_to_list_input_string(self):
        x = fe.util.to_list("a")
        self.assertEqual(x, ["a"])

    def test_to_list_input_list(self):
        x = fe.util.to_list([None, "2", (3, 4), {5, 6}, {"7":7, "8":8}, [9]])
        self.assertEqual(x, [None, "2", (3, 4), {5, 6}, {"7":7, "8":8}, [9]])

    def test_to_list_input_tuple(self):
        x = fe.util.to_list((None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]))
        self.assertEqual(x, [None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]])

    def test_to_list_input_set(self):
        x = fe.util.to_list({3,4})
        self.assertEqual(x, [3,4])

    def test_to_list_input_dict(self):
        x = fe.util.to_list({"ans": 1})
        self.assertEqual(x, [{"ans": 1}])

class TestToSet(unittest.TestCase):
    def test_to_set_input_none(self):
        x = fe.util.to_set(None)
        self.assertEqual(x, set())

    def test_to_set_input_non_iterable(self):
        x = fe.util.to_set(0)
        self.assertEqual(x, {0})

        x = fe.util.to_set("a")
        self.assertEqual(x, {"a"})

    def test_to_set_input_list(self):
        x = fe.util.to_set([None, "2", (3, 4)])
        self.assertEqual(x, {None, "2", (3, 4)})

    def test_to_set_input_tuple(self):
        x = fe.util.to_set((None, "2", (3, 4)))
        self.assertEqual(x, {None, "2", (3, 4)})

    def test_to_set_input_set(self):
        x = fe.util.to_set({3,4})
        self.assertEqual(x, {3,4})

class TestNonContext(unittest.TestCase):
    def test_non_context_syntax_work(self):
        a = 5
        with fe.util.NonContext():
            a = a + 37
            self.assertEqual(a, 42)

class TestSuppressor(unittest.TestCase):
    def test_suppressor(self):
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            with fe.util.Suppressor():
                print("hello world")
            log = fake_stdout.getvalue()
            self.assertEqual(log, '')

class TestTimer(unittest.TestCase):
    def test_timer_as_context_manager(self):
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            with fe.util.Timer():
                time.sleep(1)
            context = fake_stdout.getvalue()  # Tasks took ? seconds
        print(context)
        exec_time = float(context.split(" ")[2])
        self.assertTrue(abs(exec_time - 1) < 0.005)

    def test_timer_as_decorator(self):
        @fe.util.Timer("T2")
        def func():
            time.sleep(1)

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            func()
            context = fake_stdout.getvalue() # T2 took ? seconds

        task_name = context.split(" ")[0]
        self.assertEqual(task_name, "T2")

        exec_time = float(context.split(" ")[2])
        self.assertTrue(abs(exec_time - 1) < 0.005)


class TestDraw(unittest.TestCase):
    def test_draw_stdout(self):
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            fe.util.draw()
            log = fake_stdout.getvalue()

        expected = ('    ______           __  ______     __  _                 __            \n'
                    '   / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____\n'
                    '  / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \\/ __ `/ __/ __ \\/ ___/\n'
                    ' / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    \n'
                    '/_/    \\__,_/____/\\__/_____/____/\\__/_/_/ /_/ /_/\\__,_/\\__/\\____/_/     \n'
                    '                                                                        \n'
                    '\n')

        self.assertEqual(log, expected)

class TestPrettifyMetricName(unittest.TestCase):
    def test_prettify_metric_name(self):
        x = fe.util.prettify_metric_name("myUgly_loss")
        self.assertEqual(x, "My Ugly Loss")

    def test_prettify_metric_name_all_capital(self):
        x = fe.util.prettify_metric_name("ABCD")
        self.assertEqual(x, "Abcd")

    def test_prettify_metric_name_two_space(self):
        x = fe.util.prettify_metric_name("hello  world")
        self.assertEqual(x, "Hello World")

class TestStripSuffix(unittest.TestCase):
    def test_strip_suffix_match(self):
        x = fe.util.strip_suffix("astring.json", ".json")
        self.assertEqual(x, "astring")

    def test_strip_suffix_not_match(self):
        x = fe.util.strip_suffix("astring.json", ".yson")
        self.assertEqual(x, "astring.json")

class TestStripPrefix(unittest.TestCase):
    def test_strip_prefix_match(self):
        x = fe.util.strip_prefix("astring.json", "ast")  # "ring.json"
        self.assertEqual(x, "ring.json")

    def test_strip_prefix_not_match(self):
        x = fe.util.strip_prefix("astring.json", "asa")
        self.assertEqual(x, "astring.json")

class TestGetType(unittest.TestCase):
    def test_get_type_np(self):
        x = fe.util.get_type(np.ones((10, 10), dtype='int32'))
        self.assertEqual(x, 'int32')

        x = fe.util.get_type(np.ones((10, 10), dtype=np.float32))
        self.assertEqual(x, 'float32')

    def test_get_type_tf(self):
        x = fe.util.get_type(tf.ones((10, 10), dtype='float16'))
        self.assertEqual(x, "<dtype: 'float16'>")

        x = fe.util.get_type(tf.Variable([1,2,3], dtype=tf.int64))
        self.assertEqual(x, "<dtype: 'int64'>")

    def test_get_type_torch(self):
        x = fe.util.get_type(torch.ones((10, 10)).type(torch.float))
        self.assertEqual(x, "torch.float32")

        x = fe.util.get_type(torch.tensor((10, 10), dtype=torch.int16))
        self.assertEqual(x, "torch.int16")

    def test_get_type_list(self):
        x = fe.util.get_type([np.ones((10,10)) for i in range(4)])
        self.assertEqual(x, "List[float64]")

        x = fe.util.get_type([1, "a"]) # "List[int]"
        self.assertEqual(x, "List[int]")

        x = fe.util.get_type([[[1]]]) # "List[List[List[int]]]"
        self.assertEqual(x, "List[List[List[int]]]")

    def test_get_type_tuple(self):
        x = fe.util.get_type((tf.ones((10, 10), dtype='float16'), ))
        self.assertEqual(x, "List[<dtype: 'float16'>]")

class TestGetShape(unittest.TestCase):
    def test_get_shape_np_dimension_match(self):
        x = fe.util.get_shape(np.ones((12,22,11)))
        self.assertEqual(x, [12, 22, 11])

        x = fe.util.get_shape([np.ones((12,22,11))])
        self.assertEqual(x, [None, 12, 22, 11])

        x = fe.util.get_shape([np.ones((12,22,11)), np.ones((12,22,11))])
        self.assertEqual(x, [None, 12, 22, 11])

        x = fe.util.get_shape([np.ones((12,22,11)), np.ones((12, 22, 4))])
        self.assertEqual(x, [None, 12, 22, None])

        x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5, 4))])
        self.assertEqual(x, [None, None, None, None])

    def test_get_shape_np_dimension_mismatch(self):
        x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5))])
        self.assertEqual(x, [None])


class TestPadBatch(unittest.TestCase):
    def test_pad_batch_pad_one_entry(self):
        data = [{"x": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
        fe.util.pad_batch(data, pad_value=0)
        obj = [{"x": np.array([[1.,1.], [1., 1.], [0.,0.]]), "y": 8},
                {"x": np.array([[1.,0.], [1., 0.], [1.,0.]]), "y": 4}]

        self.assertTrue(is_equal(data, obj))

    def test_pad_batch_pad_all_entry(self):
        data = [{"x": np.ones((3, 1)), "y": np.ones((1, 1))}, {"x": np.ones((2, 2)), "y": np.ones((1, 3))}]
        fe.util.pad_batch(data, pad_value=0)
        obj = [{"x": np.array([[1.,0.], [1., 0.], [1.,0.]]), "y": np.array([[1., 0., 0.]])},
               {"x": np.array([[1.,1.], [1., 1.], [0.,0.]]), "y": np.array([[1., 1., 1.]])}]

        self.assertTrue(is_equal(data, obj))

    def test_pad_batch_different_different_key_assertion(self):
        data = [{"x1": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
        with self.assertRaises(AssertionError):
            fe.util.pad_batch(data, pad_value=0)

    def test_pad_batch_different_rank_mismatch_assertion(self):
        data = [{"x1": np.ones((2, 2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
        with self.assertRaises(AssertionError):
            fe.util.pad_batch(data, pad_value=0)

class TestPadData(unittest.TestCase):
    
