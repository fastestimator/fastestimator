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
    """A layer for cropping along height and width dimensions.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    ```python
    x = torch.tensor(list(range(100))).view((1,1,10,10))
    m = fe.layers.pytorch.Cropping2D(3)
    y = m.forward(x)  # [[[[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56], [63, 64, 65, 66]]]]
    m = fe.layers.pytorch.Cropping2D((3, 4))
    y = m.forward(x)  # [[[[34, 35], [44, 45], [54, 55], [64, 65]]]]
    m = fe.layers.pytorch.Cropping2D(((1, 4), 4))
    y = m.forward(x)  # [[[[14, 15], [24, 25], [34, 35], [44, 45], [54, 55]]]]
    ```

    Args:
        cropping: Height and width cropping parameters. If a single int 'n' is specified, then the width and height of
            the input will both be reduced by '2n', with 'n' coming off of each side of the input. If a tuple ('h', 'w')
            is provided, then the height and width of the input will be reduced by '2h' and '2w' respectively, with 'h'
            and 'w' coming off of each side of the input. If a tuple like (('h1', 'h2'), ('w1', 'w2')) is provided, then
            'h1' will be removed from the top, 'h2' from the bottom, 'w1' from the left, and 'w2' from the right
            (assuming the top left corner as the 0,0 origin).

    Raises:
        ValueError: If `cropping` has an unacceptable data type.
    """
    def __init__(self, cropping: Union[int, Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int,
                                                                                                int]]]] = 0) -> None:
        super().__init__()

        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping), (cropping, cropping))
        elif hasattr(cropping, '__len__'):
            if len(cropping) != 2:
                raise ValueError(f"'cropping' should have two elements, but found {len(cropping)}")
            if isinstance(cropping[0], int):
                height_cropping = (cropping[0], cropping[0])
            elif hasattr(cropping[0], '__len__') and len(cropping[0]) == 2:
                height_cropping = (cropping[0][0], cropping[0][1])
            else:
                raise ValueError(f"'cropping' height should be an int or tuple of ints, but found {cropping[0]}")
            if isinstance(cropping[1], int):
                width_cropping = (cropping[1], cropping[1])
            elif hasattr(cropping[1], '__len__') and len(cropping[1]) == 2:
                width_cropping = (cropping[1][0], cropping[1][1])
            else:
                raise ValueError(f"'cropping' width should be an int or tuple of ints, but found {cropping[1]}")
            self.cropping = (height_cropping, width_cropping)
        else:
            raise ValueError(
                "cropping` should be either an int, a tuple of 2 ints or a tuple of two tuple of 2 ints. Found: " +
                str(cropping))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:-self.cropping[1][1]]
