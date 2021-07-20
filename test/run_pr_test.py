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
import os
import unittest

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

loader = unittest.TestLoader()
test_dir = os.path.join(__file__, "..", "PR_test")
suite = loader.discover(test_dir)

runner = unittest.TextTestRunner()
res = runner.run(suite)

if not res.wasSuccessful:
    raise ValueError("not all tests were successfully executed (pass or fail)")

if res.failures or res.errors:
    raise ValueError("not all tests passed")
