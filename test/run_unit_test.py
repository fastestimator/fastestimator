import os
import sys
import unittest

loader = unittest.TestLoader()
test_dir = os.path.join(__file__, "..", "unit_test")
suite = loader.discover(test_dir)

runner = unittest.TextTestRunner()
res = runner.run(suite)

if not res.wasSuccessful:
    raise ValueError("not all tests were successfully executed (pass or fail)")

if res.failures:
    raise ValueError("not all tests passed")

