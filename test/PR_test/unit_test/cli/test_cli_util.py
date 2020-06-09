import unittest

from fastestimator.cli.cli_util import parse_cli_to_dictionary


class TestCliUtil(unittest.TestCase):
    def test_parse_cli_to_dictionary(self):
        a = parse_cli_to_dictionary(["--epochs", "5", "--test", "this", "--lr", "0.74"])
        self.assertEqual(a, {'epochs': 5, 'test': 'this', 'lr': 0.74})

    def test_parse_cli_to_dictionary_no_key(self):
        a = parse_cli_to_dictionary(["abc", "def"])
        self.assertEqual(a, {})

    def test_parse_cli_to_dictionary_no_value(self):
        a = parse_cli_to_dictionary(["--abc", "--def"])
        self.assertEqual(a, {"abc": "", "def": ""})

    def test_parse_cli_to_dictionary_consecutive_value(self):
        a = parse_cli_to_dictionary(["--abc", "hello", "world"])
        self.assertEqual(a, {"abc": "helloworld"})
