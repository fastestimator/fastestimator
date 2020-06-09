import os
import unittest

from fastestimator.cli.train import _get_estimator


class TestGetEstimator(unittest.TestCase):
    """
    This integration test covers:
    * fe.cli.train._get_estimator
    * fe.cli.cli_util.parse_cli_to_dictionary
    """
    def test_get_estimator_no_hyperparameters_json(self):
        args = {
            "entry_point": os.path.join(os.path.abspath(__file__), "..", "resources", "sample.py"),
            'hyperparameters_json': None
        }

        unknown = ["--epochs", "32", "--hello", "world"]

        est = _get_estimator(args=args, unknown=unknown)
        self.assertEqual(est, {"epochs": 32, "hello": "world"})

    def test_get_estimator_hyperparameters_json(self):
        args = {
            "entry_point": os.path.join(os.path.abspath(__file__), "..", "resources", "sample.py"),
            'hyperparameters_json': os.path.join(os.path.abspath(__file__), "..", "resources", "hyperparameter.json")
        }

        unknown = ["--epochs", "32", "--hello", "world"]

        est = _get_estimator(args=args, unknown=unknown)
        self.assertEqual(est, {"epochs": 32, "hello": "world", "int": 10, "list": "[1, 2, 3]", "string": "apple"})
