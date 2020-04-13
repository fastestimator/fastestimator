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
from typing import Tuple, Union

import torch
import torch.nn as nn


class Cropping2D(nn.Module):
    """Class for cropping along height and width dimensions

    Args:
        cropping: height and width cropping.

    Raises:
        ValueError: If cropping has unacceptable data type.
    """
    def __init__(self, cropping: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = 0) -> None:
        super().__init__()

        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping), (cropping, cropping))
        elif hasattr(cropping, '__len__'):
            if len(cropping) != 2:
                raise ValueError("cropping should have two elements. Found" + str(cropping))
            if isinstance(cropping[0], int) and isinstance(cropping[1], int):
                height_cropping = (cropping[0], cropping[0])
                width_cropping = (cropping[1], cropping[1])
            elif len(cropping[0]) == 2 and len(cropping[1]) == 2:
                height_cropping = (cropping[0][0], cropping[0][1])
                width_cropping = (cropping[1][0], cropping[1][1])
            else:
                raise ValueError(
                    "cropping` should be either an int, a tuple of 2 ints or a tuple of two tuple of 2 ints. Found: " +
                    str(cropping))

            self.cropping = (height_cropping, width_cropping)
        else:
            raise ValueError(
                "cropping` should be either an int, a tuple of 2 ints or a tuple of two tuple of 2 ints. Found: " +
                str(cropping))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:-self.cropping[1][1]]
