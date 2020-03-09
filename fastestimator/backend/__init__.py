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
from fastestimator.backend.binary_crossentropy import binary_crossentropy
from fastestimator.backend.categorical_crossentropy import categorical_crossentropy
from fastestimator.backend.feed_forward import feed_forward
from fastestimator.backend.get_gradient import get_gradient
from fastestimator.backend.load_model import load_model
from fastestimator.backend.reduce_loss import reduce_loss
from fastestimator.backend.save_model import save_model
from fastestimator.backend.sparse_categorical_crossentropy import sparse_categorical_crossentropy
from fastestimator.backend.to_number import to_number
from fastestimator.backend.to_shape import to_shape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.to_type import to_type
from fastestimator.backend.update_model import update_model
