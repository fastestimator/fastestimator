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
    def test_cli_main_run_train(self):
        with patch('fastestimator.cli.train.train') as fake:
            fe.cli.run(["train", "example_entry.py", "--epochs", "3", "--batch_size", "64"])
            args, unknown = fake.call_args[0]

            with self.subTest('args["mode"]'):
                self.assertEqual(args["mode"], "train")

            with self.subTest('args["entry_point"]'):
                self.assertEqual(args["entry_point"], "example_entry.py")

            with self.subTest('args["hyperparameters_json"]'):
                self.assertEqual(args["hyperparameters_json"], None)

            with self.subTest('args["warmup"]'):
                self.assertEqual(args["warmup"], "true")

            with self.subTest("unknown"):
                self.assertEqual(unknown, ['--epochs', '3', '--batch_size', '64'])

    def test_cli_main_run_test(self):
        with patch('fastestimator.cli.train.test') as fake:
            fe.cli.run(["test", "example_entry.py", "--randon", "200"])
            args, unknown = fake.call_args[0]

            with self.subTest('args["mode"]'):
                self.assertEqual(args["mode"], "test")

            with self.subTest('args["entry_point"]'):
                self.assertEqual(args["entry_point"], "example_entry.py")

            with self.subTest('args["hyperparameters_json"]'):
                self.assertEqual(args["hyperparameters_json"], None)

            with self.subTest("unknown"):
                self.assertEqual(unknown, ["--randon", "200"])

    def test_cli_main_run_logs(self):
        with patch('fastestimator.cli.logs.logs') as fake:
            fe.cli.run(["logs", "example_entry.py", "-b", "-v"])
            args, unknown = fake.call_args[0]

            with self.subTest('args["mode"]'):
                self.assertEqual(args["mode"], "logs")

            with self.subTest('args["log_dir"]'):
                self.assertEqual(args["log_dir"], "example_entry.py")

            with self.subTest('args["recursive"]'):
                self.assertEqual(args["recursive"], False)

            with self.subTest('args["ignore"]'):
                self.assertEqual(args["ignore"], None)

            with self.subTest('args["smooth"]'):
                self.assertEqual(args["smooth"], 1)

            with self.subTest('args["pretty_names"]'):
                self.assertEqual(args["pretty_names"], False)

            with self.subTest('args["smooth"]'):
                self.assertEqual(args["smooth"], 1)

            with self.subTest('args["share_legend"]'):
                self.assertEqual(args["share_legend"], True)

            with self.subTest('args["save"]'):
                self.assertEqual(args["save"], False)

            with self.subTest('args["save_dir"]'):
                self.assertEqual(args["save_dir"], None)

            with self.subTest("unknown"):
                self.assertEqual(unknown, ["-b", "-v"])
