import os
import sys
import unittest

loader = unittest.TestLoader()
test_dir = os.path.join(__file__, "..", "unit_test")
suite = loader.discover(test_dir)

runner = unittest.TextTestRunner()
res = runner.run(suite)
sys.exit(0 if res.wasSuccessful else 1)
