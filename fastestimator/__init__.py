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
import tensorflow as tf

from fastestimator.estimator import Estimator
from fastestimator.network import Network, build
from fastestimator.pipeline import Pipeline
from fastestimator.record_writer import RecordWriter
from fastestimator.util.util import get_num_devices

__version__ = '1.0-beta2'

if get_num_devices() > 1:
    distribute_strategy = tf.distribute.MirroredStrategy()
else:
    distribute_strategy = None
