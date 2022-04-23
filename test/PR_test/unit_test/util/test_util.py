# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.client import device_lib

import fastestimator as fe
import fastestimator.util.cli_util
from fastestimator.test.unittest_util import is_equal


class TestParseStringToPython(unittest.TestCase):
    def test_parse_string_to_python_positive_int(self):
        x = fastestimator.util.cli_util.parse_string_to_python("5")
        self.assertEqual(x, 5)

    def test_parse_string_to_python_negative_float(self):
        x = fastestimator.util.cli_util.parse_string_to_python("-1.5")
        self.assertEqual(x, -1.5)

    def test_parse_string_to_python_list(self):
        x = fastestimator.util.cli_util.parse_string_to_python("[5, 4, 0.3]")
        self.assertTrue(x, [5, 4, 0.3])

    def test_parse_string_to_python_list_over_mixed_entries(self):
        x = fastestimator.util.cli_util.parse_string_to_python("[5, [1,2,3]]")
        self.assertTrue(x, [5, [1, 2, 3]])

    def test_parse_string_to_python_dict(self):
        x = fastestimator.util.cli_util.parse_string_to_python("{'a': 5, 'b': 7}")
        self.assertEqual(x, {'a': 5, 'b': 7})

    def test_parse_string_to_python_dict_over_mixed_entries(self):
        x = fastestimator.util.cli_util.parse_string_to_python("{'a':5, 'b':[1,2,3]}")
        self.assertEqual(x, {'a': 5, 'b': [1, 2, 3]})


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
        x = fe.util.to_list([None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]])
        self.assertEqual(x, [None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]])

    def test_to_list_input_tuple(self):
        x = fe.util.to_list((None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]))
        self.assertEqual(x, [None, "2", (3, 4), {5, 6}, {"7": 7, "8": 8}, [9]])

    def test_to_list_input_set(self):
        x = fe.util.to_list({3, 4})
        self.assertEqual(x, [3, 4])

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
        x = fe.util.to_set({3, 4})
        self.assertEqual(x, {3, 4})


class TestParamToRange(unittest.TestCase):
    def test_param_to_range_int(self):
        x = fe.util.param_to_range(3)
        self.assertEqual(x, (-3, 3))

    def test_param_to_range_float(self):
        x = fe.util.param_to_range(2.3)
        self.assertEqual(x, (-2.3, 2.3))

    def test_param_to_range_neg_int(self):
        x = fe.util.param_to_range(-3)
        self.assertEqual(x, (-3, 3))

    def test_param_to_range_neg_float(self):
        x = fe.util.param_to_range(-2.3)
        self.assertEqual(x, (-2.3, 2.3))

    def test_param_to_range_int_tuple(self):
        x = fe.util.param_to_range((2, 3))
        self.assertEqual(x, (2, 3))

    def test_param_to_range_float_tuple(self):
        x = fe.util.param_to_range((1.2, 3.3))
        self.assertEqual(x, (1.2, 3.3))


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
            with patch('tensorflow.print', new=print):
                with fe.util.Timer():
                    time.sleep(1)
                context = fake_stdout.getvalue()  # Tasks took ? seconds
        exec_time = float(context.split(" ")[2])
        self.assertTrue(abs(exec_time - 1) < 0.1)

    def test_timer_as_decorator(self):
        @fe.util.Timer("T2")
        def func():
            time.sleep(1)

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            with patch('tensorflow.print', new=print):
                func()
                context = fake_stdout.getvalue()  # T2 took ? seconds

        with self.subTest("test the printed name"):
            task_name = context.split(" ")[0]
            self.assertEqual(task_name, "T2")

        with self.subTest("test recorded time"):
            exec_time = float(context.split(" ")[2])
            self.assertTrue(abs(exec_time - 1) < 0.1)


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

        x = fe.util.get_type(tf.Variable([1, 2, 3], dtype=tf.int64))
        self.assertEqual(x, "<dtype: 'int64'>")

    def test_get_type_torch(self):
        x = fe.util.get_type(torch.ones((10, 10)).type(torch.float))
        self.assertEqual(x, "torch.float32")

        x = fe.util.get_type(torch.tensor((10, 10), dtype=torch.int16))
        self.assertEqual(x, "torch.int16")

    def test_get_type_list(self):
        x = fe.util.get_type([np.ones((10, 10)) for i in range(4)])
        self.assertEqual(x, "List[float64]")

        x = fe.util.get_type([1, "a"])  # "List[int]"
        self.assertEqual(x, "List[int]")

        x = fe.util.get_type([[[1]]])  # "List[List[List[int]]]"
        self.assertEqual(x, "List[List[List[int]]]")

    def test_get_type_tuple(self):
        x = fe.util.get_type((tf.ones((10, 10), dtype='float16'), ))
        self.assertEqual(x, "List[<dtype: 'float16'>]")


class TestGetShape(unittest.TestCase):
    def test_get_shape_np_dimension_match(self):
        x = fe.util.get_shape(np.ones((12, 22, 11)))
        self.assertEqual(x, [12, 22, 11])

        x = fe.util.get_shape([np.ones((12, 22, 11))])
        self.assertEqual(x, [None, 12, 22, 11])

        x = fe.util.get_shape([np.ones((12, 22, 11)), np.ones((12, 22, 11))])
        self.assertEqual(x, [None, 12, 22, 11])

        x = fe.util.get_shape([np.ones((12, 22, 11)), np.ones((12, 22, 4))])
        self.assertEqual(x, [None, 12, 22, None])

        x = fe.util.get_shape([np.ones((12, 22, 11)), np.ones((18, 5, 4))])
        self.assertEqual(x, [None, None, None, None])

    def test_get_shape_np_dimension_mismatch(self):
        x = fe.util.get_shape([np.ones((12, 22, 11)), np.ones((18, 5))])
        self.assertEqual(x, [None])


class TestParseModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modes = {"train", "eval", "test", "infer"}
        cls.n_modes = {"!train", "!eval", "!test", "!infer"}
        cls.n_modes_map = {
            "!train": {"eval", "test", "infer"},
            "!eval": {"train", "test", "infer"},
            "!test": {"train", "eval", "infer"},
            "!infer": {"train", "eval", "test"}
        }

    def test_parse_modes_single_mode_direct(self):
        for mode in self.modes:
            with self.subTest(modes={mode}):
                self.assertEqual(fe.util.parse_modes({mode}), {mode})

    def test_parse_modes_single_mode_negation(self):
        for mode in self.n_modes:
            with self.subTest(modes={mode}):
                self.assertEqual(fe.util.parse_modes({mode}), self.n_modes_map[mode])

    def test_parse_modes_mutilple_mode_direct(self):
        modes = [{"train", "eval"}, {"train", "eval", "test"}, {"train", "eval", "test", "infer"}]
        for mode in modes:
            with self.subTest(modes=mode):
                self.assertEqual(fe.util.parse_modes(mode), mode)

    def test_parse_modes_mutilple_mode_negation(self):
        modes = [{"!train", "!eval"}, {"!train", "!eval", "!test"}, {"!train", "!eval", "!test", "!infer"}]
        anss = [{"test", "infer"}, {"infer"}, set()]
        for mode, ans in zip(modes, anss):
            with self.subTest(modes=mode):
                self.assertEqual(fe.util.parse_modes(mode), ans)

    def test_parse_modes_invalid_mode(self):
        with self.assertRaises(AssertionError):
            fe.util.parse_modes({"wrong_mode"})

    def test_parse_modes_empty_set(self):
        self.assertEqual(fe.util.parse_modes(set()), set())

    def test_parse_modes_mix_true_and_negation(self):
        with self.assertRaises(AssertionError):
            fe.util.parse_modes({"train", "!eval"})


class TestPadBatch(unittest.TestCase):
    def test_pad_batch_pad_one_entry(self):
        data = [{"x": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
        fe.util.pad_batch(data, pad_value=0)
        obj = [{
            "x": np.array([[1., 1.], [1., 1.], [0., 0.]]), "y": 8
        }, {
            "x": np.array([[1., 0.], [1., 0.], [1., 0.]]), "y": 4
        }]

        self.assertTrue(is_equal(data, obj))

    def test_pad_batch_pad_all_entry(self):
        data = [{"x": np.ones((3, 1)), "y": np.ones((1, 1))}, {"x": np.ones((2, 2)), "y": np.ones((1, 3))}]
        fe.util.pad_batch(data, pad_value=0)
        obj = [{
            "x": np.array([[1., 0.], [1., 0.], [1., 0.]]), "y": np.array([[1., 0., 0.]])
        }, {
            "x": np.array([[1., 1.], [1., 1.], [0., 0.]]), "y": np.array([[1., 1., 1.]])
        }]

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
    def test_pad_data_target_shape_all_dimension_larger(self):
        x = np.ones((1, 2))
        x = fe.util.pad_data(x, target_shape=(3, 3), pad_value=-2)
        y = np.array([[1, 1, -2], [-2, -2, -2], [-2, -2, -2]])
        self.assertTrue(is_equal(x, y))

    def test_pad_data_target_shape_has_higher_rank(self):
        x = np.ones((1, 2))
        with self.assertRaises(ValueError):
            x = fe.util.pad_data(x, target_shape=(3, 3, 3), pad_value=-2)

    def test_pad_data_target_shape_has_smaller_rank(self):
        x = np.ones((1, 2))
        with self.assertRaises(ValueError):
            x = fe.util.pad_data(x, target_shape=(1, ), pad_value=-2)

    def test_pad_data_target_shape_all_dimension_smaller(self):
        x = np.ones((3, 3))
        with self.assertRaises(ValueError):
            x = fe.util.pad_data(x, target_shape=(1, 2), pad_value=-2)

    def test_pad_data_target_shape_some_dimension_smaller(self):
        x = np.ones((3, 3))
        with self.assertRaises(ValueError):
            x = fe.util.pad_data(x, target_shape=(1, 4), pad_value=-2)


class TestIsNumber(unittest.TestCase):
    def test_is_number_pos_float(self):
        x = fe.util.is_number("13.7")
        self.assertTrue(x)

    def test_is_number_neg_float(self):
        x = fe.util.is_number("-8.64")
        self.assertTrue(x)

    def test_is_number_pos_scientific_expression(self):
        x = fe.util.is_number("2.5e-10")
        self.assertTrue(x)

    def test_is_number_neg_scientific_expression(self):
        x = fe.util.is_number("-2.5e5")
        self.assertTrue(x)

    def test_is_number_string(self):
        x = fe.util.is_number("apple")
        self.assertFalse(x)

    def test_is_number_string_with_number(self):
        x = fe.util.is_number("123hello")
        self.assertFalse(x)

    def test_is_number_empty_string(self):
        x = fe.util.is_number("")
        self.assertFalse(x)


class TestDefaultKeyDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dict = fe.util.DefaultKeyDict(default=lambda x: x + x, a=4, b=6)

    def test_default_key_dict_use_existed_key(self):
        self.assertEqual(self.test_dict["a"], 4)

    def test_default_key_dict_use_non_existed_key(self):
        self.assertEqual(self.test_dict[10], 20)

    def test_default_key_dict_write_value_on_existed_key(self):
        self.test_dict["a"] = -100
        self.assertEqual(self.test_dict["a"], -100)

    def test_default_key_dict_write_value_on_non_existed_key(self):
        self.test_dict["c"] = "hello"
        self.assertEqual(self.test_dict["c"], "hello")


class TestGetNumDevices(unittest.TestCase):
    def test_get_num_devices(self):
        x = fe.util.get_num_devices()
        local_device_protos = device_lib.list_local_devices()
        gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        ans = max(1, len(gpu_list))
        self.assertEqual(x, ans)


class TestGetBatchSize(unittest.TestCase):
    def test_get_batch_size_np(self):
        data = {"a": np.ones([3, 4, 5])}
        batch_size = fe.util.get_batch_size(data)
        self.assertEqual(batch_size, 3)

    def test_get_batch_size_tf(self):
        data = {"a": tf.ones([3, 4, 5])}
        batch_size = fe.util.get_batch_size(data)
        self.assertEqual(batch_size, 3)

    def test_get_batch_size_torch(self):
        data = {"a": torch.ones([3, 4, 5])}
        batch_size = fe.util.get_batch_size(data)
        self.assertEqual(batch_size, 3)

    def test_get_batch_size_np_different_shape(self):
        data = {"a": np.ones([3, 4, 5]), "b": np.ones([1, 2])}
        with self.assertRaises(AssertionError):
            batch_size = fe.util.get_batch_size(data)

    def test_get_batch_size_list(self):
        data = [np.ones([3, 4, 5])]
        with self.assertRaises(AssertionError):
            batch_size = fe.util.get_batch_size(data)

    def test_get_batch_size_dict_with_all_irrelevent_value(self):
        data = {"a": 1, "b": "hello"}
        with self.assertRaises(AssertionError):
            batch_size = fe.util.get_batch_size(data)


class TestToNumber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = np.array([1, 2, 3])
        cls.t = tf.constant([1, 2, 3])
        cls.p = torch.tensor([1, 2, 3])

    def test_to_number_np_value(self):
        self.assertTrue(np.allclose(fe.util.util.to_number(self.n), self.n))

    def test_to_number_np_type(self):
        self.assertEqual(type(fe.util.util.to_number(self.n)), np.ndarray)

    def test_to_number_tf_value(self):
        self.assertTrue(np.allclose(fe.util.util.to_number(self.t), self.n))

    def test_to_number_tf_type(self):
        self.assertEqual(type(fe.util.util.to_number(self.t)), np.ndarray)

    def test_to_number_torch_value(self):
        self.assertTrue(np.allclose(fe.util.util.to_number(self.p), self.n))

    def test_to_number_torch_type(self):
        self.assertEqual(type(fe.util.util.to_number(self.p)), np.ndarray)
