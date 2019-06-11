import copy
import io
from unittest import TestCase
from unittest.mock import patch, MagicMock

from .parse_logs import *

mock_good_file_path = '/good/file1.txt'
mock_good_file_contents = 'FastEstimator: Model artifact will be saved in /opt/ml/model\nFastEstimator: Found 10092 examples for training and 2523 for validation in /opt/ml/input/data/training\nFastEstimator-Train:step: 0; train_loss: 12.027558; lr: 0.000200; example/sec: 0.000000;\nFastEstimator-Train:step: 100; train_loss: 2.565781; lr: 0.000200; example/sec: 44.738688;\nFastEstimator-Train:step: 200; train_loss: 0.824913; lr: 0.000200; example/sec: 45.086421;\nFastEstimator-Eval:step: 281; val_loss: 0.725258; min_val_loss: 0.725258; since_best: 0; val_mask_raw_loss: -0.007752625846316934; val_image_labels_loss: 0.7330105359355609; val_mask_raw_conditionalDice: 0.007752625846316934; val_image_labels_my_binary_accuracy: 0.5194444588075081;\nFastEstimator-Train:step: 300; train_loss: 0.561318; lr: 0.000200; example/sec: 44.689092;\nFastEstimator-Train:step: 400; train_loss: 0.427389; lr: 0.000200; example/sec: 44.799198;\nFastEstimator-Train:step: 500; train_loss: 0.528405; lr: 0.000200; example/sec: 44.523727;\nFastEstimator-Eval:step: 562; val_loss: 4.125795; min_val_loss: 0.725258; since_best: 1; val_mask_raw_loss: -0.15434368319272632; val_image_labels_loss: 4.280138591821823; val_mask_raw_conditionalDice: 0.15434368319272632; val_image_labels_my_binary_accuracy: 0.5662698552840286;\nFastEstimator-Train:step: 600; train_loss: 0.686736; lr: 0.000200; example/sec: 45.055799;'
mock_good_parsed_result = {
    "train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389], [500, 0.528405],
                   [600, 0.686736]],
    "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002], [600, 0.0002]],
    "example/sec": [[0, 0.0], [100, 44.738688], [200, 45.086421], [300, 44.689092], [400, 44.799198], [500, 44.523727],
                    [600, 45.055799]], "val_loss": [[281, 0.725258], [562, 4.125795]],
    "min_val_loss": [[281, 0.725258], [562, 0.725258]], "since_best": [[281, 0.0], [562, 1.0]],
    "val_mask_raw_loss": [[281, -0.007752625846316934], [562, -0.15434368319272632]],
    "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]],
    "val_mask_raw_conditionalDice": [[281, 0.007752625846316934], [562, 0.15434368319272632]],
    "val_image_labels_my_binary_accuracy": [[281, 0.5194444588075081], [562, 0.5662698552840286]]}
mock_good_file_path_2 = '/good/file2.txt'
mock_good_file_contents_2 = 'FastEstimator: Model artifact will be saved in /opt/ml/model\nFastEstimator: Found 10092 examples for training and 2523 for validation in /opt/ml/input/data/training\nFastEstimator-Train:step: 0; train_loss: 3.038919; lr: 0.000200; example/sec: 0.000000;\nFastEstimator-Train:step: 100; train_loss: 0.887208; lr: 0.000200; example/sec: 44.704689;\nFastEstimator-Train:step: 200; train_loss: 0.541103; lr: 0.000200; example/sec: 44.553714;\nFastEstimator-Eval:step: 281; val_loss: 3.122921; min_val_loss: 3.122921; since_best: 0; val_mask_raw_loss: -0.08894996416088866; val_image_labels_loss: 3.2118707436654303; val_mask_raw_conditionalDice: 0.08894996416088866; val_image_labels_my_binary_accuracy: 0.5757936653577619;\nFastEstimator-Train:step: 300; train_loss: 1.814015; lr: 0.000200; example/sec: 45.014834;\nFastEstimator-Train:step: 400; train_loss: 0.485160; lr: 0.000200; example/sec: 44.368620;\nFastEstimator-Train:step: 500; train_loss: 0.375250; lr: 0.000200; example/sec: 44.782828;\nFastEstimator-Eval:step: 562; val_loss: 1.244561; min_val_loss: 1.244561; since_best: 0; val_mask_raw_loss: -0.13716965285970623; val_image_labels_loss: 1.3817303948311341; val_mask_raw_conditionalDice: 0.13716965285970623; val_image_labels_my_binary_accuracy: 0.624603189734949;\nFastEstimator-Train:step: 600; train_loss: 0.481111; lr: 0.000200; example/sec: 44.513710;\nFastEstimator-Train:step: 700; train_loss: 0.308867; lr: 0.000200; example/sec: 44.525426;\nFastEstimator-Train:step: 800; train_loss: 0.343235; lr: 0.000200; example/sec: 44.179674;\nFastEstimator-Eval:step: 843; val_loss: 1.007462; min_val_loss: 1.007462; since_best: 0; val_mask_raw_loss: -0.12723745336374526; val_image_labels_loss: 1.1346999171293444; val_mask_raw_conditionalDice: 0.12723745336374526; val_image_labels_my_binary_accuracy: 0.5924603325625261;\nFastEstimator-Train:step: 900; train_loss: 0.327094; lr: 0.000200; example/sec: 44.503702;\nFastEstimator-Train:step: 1000; train_loss: 0.555805; lr: 0.000200; example/sec: 44.428722;\nFastEstimator-Train:step: 1100; train_loss: 2.290446; lr: 0.000200; example/sec: 44.082693;\nFastEstimator-Eval:step: 1124; val_loss: 0.877936; min_val_loss: 0.877936; since_best: 0; val_mask_raw_loss: -0.21344286083541292; val_image_labels_loss: 1.0913789242506027; val_mask_raw_conditionalDice: 0.21344286083541292; val_image_labels_my_binary_accuracy: 0.6357143003907468;\nFastEstimator-Train:step: 1200; train_loss: 0.727391; lr: 0.000200; example/sec: 44.769331;\nFastEstimator-Train:step: 1300; train_loss: 1.556982; lr: 0.000200; example/sec: 44.300033;\nFastEstimator-Train:step: 1400; train_loss: 0.563514; lr: 0.000200; example/sec: 44.115453;\nFastEstimator-Eval:step: 1405; val_loss: 1.220150; min_val_loss: 0.877936; since_best: 1; val_mask_raw_loss: -0.22651713909775328; val_image_labels_loss: 1.4466672077775002; val_mask_raw_conditionalDice: 0.22651713909775328; val_image_labels_my_binary_accuracy: 0.6488095390299956;\nFastEstimator-Train:step: 1500; train_loss: 0.704434; lr: 0.000200; example/sec: 44.554155;'
mock_good_parsed_result_2 = {
    "train_loss": [[0, 3.038919], [100, 0.887208], [200, 0.541103], [300, 1.814015], [400, 0.48516], [500, 0.37525],
                   [600, 0.481111], [700, 0.308867], [800, 0.343235], [900, 0.327094], [1000, 0.555805],
                   [1100, 2.290446], [1200, 0.727391], [1300, 1.556982], [1400, 0.563514], [1500, 0.704434]],
    "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002], [600, 0.0002],
           [700, 0.0002], [800, 0.0002], [900, 0.0002], [1000, 0.0002], [1100, 0.0002], [1200, 0.0002], [1300, 0.0002],
           [1400, 0.0002], [1500, 0.0002]],
    "example/sec": [[0, 0.0], [100, 44.704689], [200, 44.553714], [300, 45.014834], [400, 44.36862], [500, 44.782828],
                    [600, 44.51371], [700, 44.525426], [800, 44.179674], [900, 44.503702], [1000, 44.428722],
                    [1100, 44.082693], [1200, 44.769331], [1300, 44.300033], [1400, 44.115453], [1500, 44.554155]],
    "val_loss": [[281, 3.122921], [562, 1.244561], [843, 1.007462], [1124, 0.877936], [1405, 1.22015]],
    "min_val_loss": [[281, 3.122921], [562, 1.244561], [843, 1.007462], [1124, 0.877936], [1405, 0.877936]],
    "since_best": [[281, 0.0], [562, 0.0], [843, 0.0], [1124, 0.0], [1405, 1.0]],
    "val_mask_raw_loss": [[281, -0.08894996416088866], [562, -0.13716965285970623], [843, -0.12723745336374526],
                          [1124, -0.21344286083541292], [1405, -0.22651713909775328]],
    "val_image_labels_loss": [[281, 3.2118707436654303], [562, 1.3817303948311341], [843, 1.1346999171293444],
                              [1124, 1.0913789242506027], [1405, 1.4466672077775002]],
    "val_mask_raw_conditionalDice": [[281, 0.08894996416088866], [562, 0.13716965285970623], [843, 0.12723745336374526],
                                     [1124, 0.21344286083541292], [1405, 0.22651713909775328]],
    "val_image_labels_my_binary_accuracy": [[281, 0.5757936653577619], [562, 0.624603189734949],
                                            [843, 0.5924603325625261], [1124, 0.6357143003907468],
                                            [1405, 0.6488095390299956]]}
mock_empty_file_path = '/bad/empty.txt'
mock_empty_file_contents = ''
mock_fake_file_path = '/bad/fake.txt'
mock_bad_file_path_missing_step = '/bad/file.txt'
mock_bad_file_contents_missing_step = 'FastEstimator: Model artifact will be saved in /opt/ml/model\nFastEstimator: Found 10092 examples for training and 2523 for validation in /opt/ml/input/data/training\nFastEstimator-Train:step: 0; train_loss: 12.027558; lr: 0.000200; example/sec: 0.000000;\nFastEstimator-Train:step: 100; train_loss: 2.565781; lr: 0.000200; example/sec: 44.738688;\nFastEstimator-Train:step: 200; train_loss: 0.824913; lr: 0.000200; example/sec: 45.086421;\nFastEstimator-Eval:step: 281; val_loss: 0.725258; min_val_loss: 0.725258; since_best: 0; val_mask_raw_loss: -0.007752625846316934; val_image_labels_loss: 0.7330105359355609; val_mask_raw_conditionalDice: 0.007752625846316934; val_image_labels_my_binary_accuracy: 0.5194444588075081;\nFastEstimator-Train:train_loss: 0.561318; lr: 0.000200; example/sec: 44.689092;\nFastEstimator-Train:step: 400; train_loss: 0.427389; lr: 0.000200; example/sec: 44.799198;\nFastEstimator-Train:step: 500; train_loss: 0.528405; lr: 0.000200; example/sec: 44.523727;\nFastEstimator-Eval:step: 562; val_loss: 4.125795; min_val_loss: 0.725258; since_best: 1; val_mask_raw_loss: -0.15434368319272632; val_image_labels_loss: 4.280138591821823; val_mask_raw_conditionalDice: 0.15434368319272632; val_image_labels_my_binary_accuracy: 0.5662698552840286;\nFastEstimator-Train:step: 600; train_loss: 0.686736; lr: 0.000200; example/sec: 45.055799;'


def load_mock_file(path):
    if mock_good_file_path == path:
        return io.StringIO(mock_good_file_contents)
    elif mock_good_file_path_2 == path:
        return io.StringIO(mock_good_file_contents_2)
    elif mock_bad_file_path_missing_step == path:
        return io.StringIO(mock_bad_file_contents_missing_step)
    elif mock_empty_file_path == path:
        return io.StringIO(mock_empty_file_contents)
    else:
        raise FileNotFoundError("[Errno 2] No such file or directory: '%s'" % path)


class TestParser(TestCase):

    # -------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------ Parser ------------------------------------------------ #
    # -------------------------------------------------------------------------------------------------------- #
    def test_parser_success(self):
        expected = mock_good_parsed_result
        # m = mock_open(read_data=mock_good_file_contents)  # Could use this instead of magic mock
        m = MagicMock(side_effect=load_mock_file)
        with patch('fastestimator.util.parse_logs.open', m):
            actual = parse_file(mock_good_file_path)
        self.assertDictEqual(actual, expected)

    def test_parser_success_2(self):
        expected = mock_good_parsed_result_2
        m = MagicMock(side_effect=load_mock_file)
        with patch('fastestimator.util.parse_logs.open', m):
            actual = parse_file(mock_good_file_path_2)
        self.assertDictEqual(actual, expected)

    def test_parser_empty_file(self):
        expected = {}
        m = MagicMock(side_effect=load_mock_file)
        with patch('fastestimator.util.parse_logs.open', m):
            actual = parse_file(mock_empty_file_path)
        self.assertDictEqual(actual, expected)

    def test_parser_file_does_not_exist(self):
        m = MagicMock(side_effect=load_mock_file)
        with patch('fastestimator.util.parse_logs.open', m):
            self.assertRaises(FileNotFoundError, parse_file, mock_fake_file_path)
        m.assert_called_once_with(mock_fake_file_path)

    def test_parser_fail_missing_step(self):
        m = MagicMock(side_effect=load_mock_file)
        with patch('fastestimator.util.parse_logs.open', m):
            self.assertRaises(AssertionError, parse_file, mock_bad_file_path_missing_step)
        m.assert_called_once_with(mock_bad_file_path_missing_step)

    # -------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------- KEY Blacklisting ------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_remove_blacklist_keys_success(self):
        expected = {"train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                                   [500, 0.528405], [600, 0.686736]],
                    "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
                           [600, 0.0002]],
                    "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]}
        blacklist = ["val_loss", "since_best", "example/sec", "min_val_loss", "val_mask_raw_loss",
                     "val_mask_raw_conditionalDice", "val_image_labels_my_binary_accuracy"]
        actual = copy.deepcopy(mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_none(self):
        expected = mock_good_parsed_result
        blacklist = None
        actual = copy.deepcopy(mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty_list(self):
        expected = mock_good_parsed_result
        blacklist = []
        actual = copy.deepcopy(mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty_set(self):
        expected = mock_good_parsed_result
        blacklist = {}
        actual = copy.deepcopy(mock_good_parsed_result)
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_empty(self):
        expected = {}
        blacklist = ["FAKE_KEY"]
        actual = {}
        remove_blacklist_keys(actual, blacklist)
        self.assertDictEqual(actual, expected)

    def test_remove_blacklist_keys_missing(self):
        expected = {"train_loss": [[0, 12.027558], [100, 2.565781], [200, 0.824913], [300, 0.561318], [400, 0.427389],
                                   [500, 0.528405], [600, 0.686736]],
                    "lr": [[0, 0.0002], [100, 0.0002], [200, 0.0002], [300, 0.0002], [400, 0.0002], [500, 0.0002],
                           [600, 0.0002]],
                    "val_image_labels_loss": [[281, 0.7330105359355609], [562, 4.280138591821823]]}
        blacklist = ["val_loss", "since_best", "example/sec", "min_val_loss", "val_mask_raw_loss",
                     "val_mask_raw_conditionalDice", "val_image_labels_my_binary_accuracy", "FAKE_KEY"]
        actual = copy.deepcopy(mock_good_parsed_result)
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
    # --------------------------------------------- Parse Files ---------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_parse_files(self):
        expected = {'file1': mock_good_parsed_result, 'file2': mock_good_parsed_result_2}

        parse_mock = MagicMock(side_effect=load_mock_file)
        graph_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.open', parse_mock):
            with patch('fastestimator.util.parse_logs.graph_metrics', graph_mock):
                parse_files([mock_good_file_path, mock_good_file_path_2], log_extension='.txt', smooth_factor=1.5,
                            save=True, save_path='/out/', ignore_metrics=None, share_legend=False, pretty_names=True)
        graph_mock.assert_called_once_with(expected, 1.5, True, '/out/', False, True)

    def test_parse_files_no_save_dir(self):
        expected = {'file1': mock_good_parsed_result, 'file2': mock_good_parsed_result_2}

        parse_mock = MagicMock(side_effect=load_mock_file)
        graph_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.open', parse_mock):
            with patch('fastestimator.util.parse_logs.graph_metrics', graph_mock):
                parse_files([mock_good_file_path, mock_good_file_path_2], log_extension='.txt', smooth_factor=1.5,
                            save=True, save_path=None, ignore_metrics=None, share_legend=False, pretty_names=True)
        graph_mock.assert_called_once_with(expected, 1.5, True, mock_good_file_path, False, True)

    def test_parse_files_no_files(self):
        self.assertRaises(AssertionError, parse_files, [])

    # -------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------- Parse Folder --------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_parse_folder(self):
        mock_dir = '/good/'
        parse_mock = MagicMock()
        is_dir_mock = MagicMock(return_value=True)
        list_mock = MagicMock(return_value=['/good/file1.log', '/good/file2.log', '/good/file3.txt', '/good/file4'])
        with patch('fastestimator.util.parse_logs.parse_files', parse_mock), patch(
                'fastestimator.util.parse_logs.os.path.isdir', is_dir_mock), patch(
            'fastestimator.util.parse_logs.os.listdir', list_mock):
            parse_folder(mock_dir, log_extension='.log', smooth_factor=2, save=True, save_path=None,
                         ignore_metrics=['lr'])
        is_dir_mock.assert_called_once_with(mock_dir)
        list_mock.assert_called_once_with(mock_dir)
        parse_mock.assert_called_once_with(['/good/file1.log', '/good/file2.log'], '.log', 2, True, None, ['lr'], True,
                                           False)

    def test_parse_folder_bad_dir(self):
        mock_dir = '/bad/'
        parse_mock = MagicMock()
        is_dir_mock = MagicMock(return_value=False)
        list_mock = MagicMock(return_value=['/good/file1.log', '/good/file2.log', '/good/file3.txt', '/good/file4'])
        with patch('fastestimator.util.parse_logs.parse_files', parse_mock), patch(
                'fastestimator.util.parse_logs.os.path.isdir', is_dir_mock), patch(
            'fastestimator.util.parse_logs.os.listdir', list_mock):
            self.assertRaises(AssertionError, parse_folder, mock_dir)
        is_dir_mock.assert_called_once_with(mock_dir)

    def test_parse_folder_empty_dir(self):
        mock_dir = '/good/'
        parse_mock = MagicMock()
        is_dir_mock = MagicMock(return_value=True)
        list_mock = MagicMock(return_value=[])
        with patch('fastestimator.util.parse_logs.parse_files', parse_mock), patch(
                'fastestimator.util.parse_logs.os.path.isdir', is_dir_mock), patch(
            'fastestimator.util.parse_logs.os.listdir', list_mock):
            parse_folder(mock_dir, log_extension='.log', smooth_factor=2, save=True, save_path=None,
                         ignore_metrics=['lr'])
        is_dir_mock.assert_called_once_with(mock_dir)
        list_mock.assert_called_once_with(mock_dir)
        parse_mock.assert_called_once_with([], '.log', 2, True, None, ['lr'], True,
                                           False)

    # -------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Graphing ----------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------- #
    def test_graph_metrics_save(self):
        metrics = {'trial1': mock_good_parsed_result, 'trial2': mock_good_parsed_result_2}
        fig_mock = MagicMock()
        dir_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.plt.savefig', fig_mock), patch(
                'fastestimator.util.parse_logs.os.makedirs', dir_mock):
            graph_metrics(metrics, 0, True, '/good/folder/trial1.txt', True, False)
        dir_mock.assert_called_once_with('/good/folder', exist_ok=True)
        fig_mock.assert_called_once_with('/good/folder/parse_logs.png', dpi=300)

    def test_graph_metrics_display(self):
        metrics = {'trial1': mock_good_parsed_result, 'trial2': mock_good_parsed_result_2}
        fig_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.plt.show', fig_mock):
            graph_metrics(metrics, 2, False, None, True, False)
        # fig_mock.assert_called_once()  # This method doesn't exist in python 3.5 :(
        assert len(fig_mock.mock_calls) == 1, "Execution failed before figure fall invoked"

    def test_graph_metrics_display_2_metrics(self):
        blacklist = ['lr', 'min_val_loss', 'since_best', 'train_loss', 'val_mask_raw_loss',
                     'val_mask_raw_conditionalDice', 'val_loss', 'val_image_labels_my_binary_accuracy']
        reduced_parse_1 = copy.deepcopy(mock_good_parsed_result)
        reduced_parse_2 = copy.deepcopy(mock_good_parsed_result_2)
        remove_blacklist_keys(reduced_parse_1, blacklist)
        remove_blacklist_keys(reduced_parse_2, blacklist)
        metrics = {'trial1': reduced_parse_1, 'trial2': reduced_parse_2}
        fig_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.plt.show', fig_mock):
            graph_metrics(metrics, 2, False, None, True, False)
        assert len(fig_mock.mock_calls) == 1, "Execution failed before figure fall invoked"


    def test_graph_metrics_display_1_metrics(self):
        blacklist = ['lr', 'min_val_loss', 'since_best', 'train_loss', 'val_mask_raw_loss',
                     'val_mask_raw_conditionalDice', 'val_loss', 'val_image_labels_my_binary_accuracy',
                     'val_image_labels_loss']
        reduced_parse_1 = copy.deepcopy(mock_good_parsed_result)
        reduced_parse_2 = copy.deepcopy(mock_good_parsed_result_2)
        remove_blacklist_keys(reduced_parse_1, blacklist)
        remove_blacklist_keys(reduced_parse_2, blacklist)
        metrics = {'trial1': reduced_parse_1, 'trial2': reduced_parse_2}
        fig_mock = MagicMock()
        with patch('fastestimator.util.parse_logs.plt.show', fig_mock):
            graph_metrics(metrics, 2, False, None, True, False)
        assert len(fig_mock.mock_calls) == 1, "Execution failed before figure fall invoked"

    def test_graph_metrics_display_0_metrics(self):
        blacklist = ['lr', 'min_val_loss', 'since_best', 'train_loss', 'val_mask_raw_loss',
                     'val_mask_raw_conditionalDice', 'val_loss', 'val_image_labels_my_binary_accuracy',
                     'val_image_labels_loss', 'example/sec']
        reduced_parse_1 = copy.deepcopy(mock_good_parsed_result)
        reduced_parse_2 = copy.deepcopy(mock_good_parsed_result_2)
        remove_blacklist_keys(reduced_parse_1, blacklist)
        remove_blacklist_keys(reduced_parse_2, blacklist)
        metrics = {'trial1': reduced_parse_1, 'trial2': reduced_parse_2}
        self.assertRaises(AssertionError, graph_metrics, metrics, 2, False, None, True, False)

    def test_graph_metrics_save_no_files(self):
        metrics = {}
        self.assertRaises(AssertionError, graph_metrics, metrics, 2, False, None, True, False)

    def test_graph_metrics_bad_smooth(self):
        metrics = {'trial1': mock_good_parsed_result, 'trial2': mock_good_parsed_result_2}
        self.assertRaises(AssertionError, graph_metrics, metrics, -1, False, None, True, False)
