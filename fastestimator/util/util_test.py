# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
import copy
from unittest import TestCase

from fastestimator.cli.cli_util import parse_cli_to_dictionary
from .util import (get_num_devices, parse_string_to_python, prettify_metric_name, remove_blacklist_keys, strip_suffix)


class TestUtil(TestCase):
    mock_good_parsed_result = {
        "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                       [500, 0.528405], [600, 0.686736]],
        "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002], [600, 0.0002]],
        "example/sec": [[0, 0.0], [100, 44.738688], [200, 45.086421], [300, 44.689092], [400, 44.799198],
                        [500, 44.523727], [600, 45.055799]],
        "val_loss": [[281, 0.725258], [562, 4.125795]],
        "min_val_loss": [[281, 0.725258], [562, 0.725258]],
        "since_best": [[281, 0.0], [562, 1.0]],
        "val_mask_raw_loss": [[281, -0.007752625846316934], [562, -0.15434368319272632]],
        "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]],
        "val_mask_raw_conditionalDice": [[281, 0.007752625846316934], [562, 0.15434368319272632]],
        "val_image_labels_my_binary_accuracy": [[281, 0.5194444588075081], [562, 0.5662698552840286]]
    }

    # -------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- GPU Count ----------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #

    def test_get_num_devices(self):
        try:
            result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            lines = [line.split() for line in result.splitlines() if line.startswith("Attached GPUs")]
            devices = int(lines[0][-1])
        except:
            devices = 1
        assert devices == get_num_devices()

    # -------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------- KEY Blacklisting ------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_remove_blacklist_keys_success(self):
        expected = {
            "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                           [500, 0.528405], [600, 0.686736]],
            "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
                   [600, 0.0002]],
            "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]
        }
        blacklist = [
            "val_loss",
            "since_best",
            "example/sec",
            "min_val_loss",
            "val_mask_raw_loss",
            "val_mask_raw_conditionalDice",
            "val_image_labels_my_binary_accuracy"
        ]
        actual = copy.deepcopy(self.mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_none(self):
        expected = self.mock_good_parsed_result
        blacklist = None
        actual = copy.deepcopy(self.mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty_list(self):
        expected = self.mock_good_parsed_result
        blacklist = []
        actual = copy.deepcopy(self.mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty_set(self):
        expected = self.mock_good_parsed_result
        blacklist = {}
        actual = copy.deepcopy(self.mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty(self):
        expected = {}
        blacklist = ["FAKE_KEY"]
        actual = {}
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_missing(self):
        expected = {
            "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                           [500, 0.528405], [600, 0.686736]],
            "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
                   [600, 0.0002]],
            "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]
        }
        blacklist = [
            "val_loss",
            "since_best",
            "example/sec",
            "min_val_loss",
            "val_mask_raw_loss",
            "val_mask_raw_conditionalDice",
            "val_image_labels_my_binary_accuracy",
            "FAKE_KEY"
        ]
        actual = copy.deepcopy(self.mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    # -------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------ Suffix ------------------------------------------------ #
    # -------------------------------------------------------------------------------------------------------- #
    def test_strip_suffix(self):
        base = "ImageNet.txt"
        suffix = ".txt"
        expected = "ImageNet"
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_empty(self):
        base = "ImageNet.txt"
        suffix = ""
        expected = "ImageNet.txt"
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_none(self):
        base = "ImageNet.txt"
        suffix = None
        expected = "ImageNet.txt"
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_full(self):
        base = "ImageNet.txt"
        suffix = "ImageNet.txt"
        expected = ""
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_wrong_suffix(self):
        base = "ImageNet.txt"
        suffix = ".tzt"
        expected = "ImageNet.txt"
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_super_suffix(self):
        base = "ImageNet.txt"
        suffix = "MImageNet.txt"
        expected = "ImageNet.txt"
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_empty_base(self):
        base = ""
        suffix = ".txt"
        expected = ""
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    def test_strip_suffix_no_base(self):
        base = None
        suffix = ".txt"
        expected = None
        actual = strip_suffix(base, suffix)
        self.assertEqual(actual, expected)

    # -------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Prettify ----------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_prettify_metric_name(self):
        base = "val_mask_raw_conditionalDice"
        expected = "Val Mask Raw Conditional Dice"
        actual = prettify_metric_name(base)
        self.assertEqual(actual, expected)

    # -------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------- Parse String --------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_parse_string_to_python_none(self):
        input_string = None
        expected = ""
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python(self):
        input_string = ""
        expected = ""
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_array(self):
        input_string = '[]'
        expected = []
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_tuple(self):
        input_string = '()'
        expected = ()
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_dict(self):
        input_string = '{}'
        expected = {}
        actual = parse_string_to_python(input_string)
        self.assertDictEqual(actual, expected)

    def test_parse_string_to_python_true(self):
        input_string = "true"
        expected = True
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_true2(self):
        input_string = "True"
        expected = True
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_false(self):
        input_string = "false"
        expected = False
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_false2(self):
        input_string = "False"
        expected = False
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_boolean_array(self):
        input_string = '[true, false, true, false]'
        expected = [True, False, True, False]
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_boolean_array2(self):
        input_string = '[True, False, True, False]'
        expected = [True, False, True, False]
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_boolean_tuple(self):
        input_string = '(True, False, True, False)'
        expected = (True, False, True, False)
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_int(self):
        input_string = "7"
        expected = 7
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_int_array(self):
        input_string = '[0, -2, 4, 8]'
        expected = [0, -2, 4, 8]
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_int_tuple(self):
        input_string = '(0, -2, 4, 8)'
        expected = (0, -2, 4, 8)
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_float(self):
        input_string = "7.5"
        expected = 7.5
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_float_array(self):
        input_string = '[0.5, -2.1, 4, 8.89]'
        expected = [0.5, -2.1, 4, 8.89]
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_float_tuple(self):
        input_string = '(0.5, -2.1, 4, 8.89)'
        expected = (0.5, -2.1, 4, 8.89)
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_string(self):
        input_string = "random string"
        expected = "random string"
        actual = parse_string_to_python(input_string)
        self.assertEqual(actual, expected)

    def test_parse_string_to_python_string_array(self):
        input_string = '["random", "string"]'
        expected = ['random', 'string']
        actual = parse_string_to_python(input_string)
        self.assertListEqual(actual, expected)

    def test_parse_string_to_python_string_tuple(self):
        input_string = '("random", "string")'
        expected = ('random', 'string')
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_string_tuple2(self):
        input_string = "('random', 'string')"
        expected = ('random', 'string')
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_nested(self):
        input_string = '("random", ["string1", 10], (True, 7.5, "string2"))'
        expected = ('random', ["string1", 10], (True, 7.5, "string2"))
        actual = parse_string_to_python(input_string)
        self.assertTupleEqual(actual, expected)

    def test_parse_string_to_python_string_dict(self):
        input_string = '{"key1":"val1","key2":"val2"}'
        expected = {"key1": "val1", "key2": "val2"}
        actual = parse_string_to_python(input_string)
        self.assertDictEqual(actual, expected)

    # -------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Parse CLI ---------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_parse_cli_to_dictionary_none(self):
        input_list = None
        expected = {}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary(self):
        input_list = []
        expected = {}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary_no_key(self):
        input_list = ["thing1", "thing2", "True", "(0,", "1)"]
        expected = {}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary_one_key_string(self):
        input_list = ["thing1", "--key1", "True", "(0,", "1)"]
        expected = {"key1": 'True(0,1)'}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary_one_key_string2(self):
        input_list = ["--key1", "True"]
        expected = {"key1": True}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary_one_key_tuple(self):
        input_list = ["thing1", "--key1", "(0,", "1)"]
        expected = {"key1": (0, 1)}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)

    def test_parse_cli_to_dictionary_two_key_tuple(self):
        input_list = ["--key1", "(0,", "1)", "--key2", "[True,", "False,", "'args']"]
        expected = {"key1": (0, 1), "key2": [True, False, 'args']}
        actual = parse_cli_to_dictionary(input_list)
        self.assertDictEqual(actual, expected)
