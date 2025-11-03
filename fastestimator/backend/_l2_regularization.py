# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import torch

Tensor = TypeVar('Tensor', bound=torch.Tensor)


def l2_regularization(model: torch.nn.Module, beta: float = 0.01) -> Tensor:
    """Calculate L2 Norm of model weights.

    l2_reg = sum(parameter**2)/2

    This method can be used with Pytorch tensors

    Args:
        model: A pytorch model
        beta: The multiplicative factor, to weight the l2 regularization loss with the input loss

    Returns:
        The L2 norm of model parameters

    Raises:
        ValueError: If `model` belongs to an unacceptable framework.
    """
    if isinstance(model, torch.nn.Module):
        l2_loss = torch.sum(torch.stack([torch.sum(p**2) / 2 for p in model.parameters() if p.requires_grad]))
    else:
        raise ValueError("Unrecognized model framework: Please make sure to pass a torch models")
    return beta * l2_loss
