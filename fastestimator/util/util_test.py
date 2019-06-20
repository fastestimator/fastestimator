import copy
from unittest import TestCase

from tensorflow.python.client import device_lib

from .util import get_num_GPU, remove_blacklist_keys, strip_suffix, prettify_metric_name


class TestUtil(TestCase):
    mock_good_parsed_result = {
        "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                       [500, 0.528405],
                       [600, 0.686736]],
        "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
               [600, 0.0002]],
        "example/sec": [[0, 0.0], [100, 44.738688], [200, 45.086421], [300, 44.689092], [400, 44.799198],
                        [500, 44.523727],
                        [600, 45.055799]], "val_loss": [[281, 0.725258], [562, 4.125795]],
        "min_val_loss": [[281, 0.725258], [562, 0.725258]], "since_best": [[281, 0.0], [562, 1.0]],
        "val_mask_raw_loss": [[281, -0.007752625846316934], [562, -0.15434368319272632]],
        "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]],
        "val_mask_raw_conditionalDice": [[281, 0.007752625846316934], [562, 0.15434368319272632]],
        "val_image_labels_my_binary_accuracy": [[281, 0.5194444588075081], [562, 0.5662698552840286]]}

    # -------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- GPU Count ----------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #

    def test_get_num_GPU(self):
        local_device_protos = device_lib.list_local_devices()
        num_gpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])

        assert num_gpu == get_num_GPU()

    # -------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------- KEY Blacklisting ------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_remove_blacklist_keys_success(self):
        expected = {
            "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                           [500, 0.528405], [600, 0.686736]],
            "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
                   [600, 0.0002]],
            "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]}
        blacklist = ["val_loss", "since_best", "example/sec", "min_val_loss", "val_mask_raw_loss",
                     "val_mask_raw_conditionalDice", "val_image_labels_my_binary_accuracy"]
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
            "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]}
        blacklist = ["val_loss", "since_best", "example/sec", "min_val_loss", "val_mask_raw_loss",
                     "val_mask_raw_conditionalDice", "val_image_labels_my_binary_accuracy", "FAKE_KEY"]
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
