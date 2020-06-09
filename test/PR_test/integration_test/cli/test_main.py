import unittest
from unittest.mock import patch

import fastestimator as fe


class TestMain(unittest.TestCase):
    """
    This integration test covers:
    * fe.cli.logs.configure_log_parser
    * fe.cli.train.configure_test_parser
    * fe.cli.train.configure_train_parser
    * fe.cli.run
    """
    def mock_fuc(self, args, unknown):
        self.args = args
        self.unknown = unknown

    def test_cli_main_run_train(self):
        with patch('fastestimator.cli.train.train', new=self.mock_fuc):
            fe.cli.run(["train", "example_entry.py", "--epochs", "3", "--batch_size", "64"])
            arg_key = "mode"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "train")

            arg_key = "entry_point"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "example_entry.py")

            arg_key = "hyperparameters_json"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], None)

            arg_key = "warmup"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "true")

            with self.subTest("unknown args"):
                self.assertEqual(self.unknown, ['--epochs', '3', '--batch_size', '64'])

    def test_cli_main_run_test(self):
        with patch('fastestimator.cli.train.test', new=self.mock_fuc):
            fe.cli.run(["test", "example_entry.py", "--randon", "200"])

            arg_key = "mode"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "test")

            arg_key = "entry_point"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "example_entry.py")

            arg_key = "hyperparameters_json"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], None)

            with self.subTest("unknown args"):
                self.assertEqual(self.unknown, ["--randon", "200"])

    def test_cli_main_run_logs(self):
        with patch('fastestimator.cli.logs.logs', new=self.mock_fuc):
            fe.cli.run(["logs", "example_entry.py", "-b", "-v"])

            arg_key = "mode"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "logs")

            arg_key = "log_dir"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], "example_entry.py")

            arg_key = "recursive"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], False)

            arg_key = "ignore"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], None)

            arg_key = "smooth"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], 1)

            arg_key = "pretty_names"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], False)

            arg_key = "smooth"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], 1)

            arg_key = "share_legend"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], True)

            arg_key = "save"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], False)

            arg_key = "save_dir"
            with self.subTest(arg_key=arg_key):
                self.assertEqual(self.args[arg_key], None)

            with self.subTest("unknown args"):
                self.assertEqual(self.unknown, ["-b", "-v"])
