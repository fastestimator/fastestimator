# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import TypeVar

import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
import torch

from fastestimator.backend._cast import cast

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def convert_tensor_precision(tensor: Tensor) -> Tensor:
    """
        Adjust the input data precision based of environment precision.

        Args:
            tensor: The input value.

        Returns:
            The precision adjusted data(16 bit for mixed precision, 32 bit otherwise).

    """
    precision = 'float32'

    if mixed_precision.global_policy().compute_dtype == 'float16':
        precision = 'float16'

    return cast(tensor, precision)
