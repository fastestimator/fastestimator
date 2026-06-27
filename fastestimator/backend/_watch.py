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
import torch


def watch(tensor: torch.Tensor) -> torch.Tensor:
    """Monitor the given `tensor` for later gradient computations.

    This method can be used with PyTorch tensors:
    ```python
    x = torch.ones((3,1,28,28))  # x.requires_grad == False
    x = fe.backend.watch(x)  # x.requires_grad == True
    ```

    Args:
        tensor: The tensor to be monitored.

    Returns:
        The `tensor` or a copy of the `tensor` which is being tracked for gradient computations. This value is only
        needed if using PyTorch as the backend.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            return tensor
        # It is tempting to just do tensor.requires_grad = True here, but that will lead to trouble
        return tensor.detach().requires_grad_(True)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
