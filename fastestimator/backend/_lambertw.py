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
import math
from typing import TypeVar

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from scipy.special import lambertw as lamw

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def lambertw(tensor: Tensor) -> Tensor:
    """Compute the k=0 branch of the Lambert W function.

    See https://en.wikipedia.org/wiki/Lambert_W_function for details. Only valid for inputs >= -1/e (approx -0.368). We
    do not check this for the sake of speed, but if an input is out of domain the return value may be random /
    inconsistent or even NaN.

    This method can be used with Numpy data:
    ```python
    n = np.array([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
    b = fe.backend.lambertw(n)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
    b = fe.backend.lambertw(t)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
    b = fe.backend.lambertw(p)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
    ```

    Args:
        tensor: The input value.

    Returns:
        The lambertw function evaluated at `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tfp.math.lambertw(tensor)
    if isinstance(tensor, torch.Tensor):
        return _torch_lambertw(tensor)
    elif isinstance(tensor, np.ndarray):
        # scipy implementation is numerically unstable at exactly -1/e, but the result should be -1.0
        return np.nan_to_num(lamw(tensor, k=0, tol=1e-6).real.astype(tensor.dtype), nan=-1.0)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def _torch_lambertw(z: torch.Tensor) -> torch.Tensor:
    """Approximate the LambertW function value using Halley iteration.

    Args:
        z: The inputs to the LambertW function.

    Returns:
        An approximation of W(z).
    """
    # Make some starting guesses in order to converge faster.
    z0 = torch.where(z < -0.2649, _taylor_approx(z), _lambertw_winitzki_approx(z))
    tolerance = 1e-6
    # Perform at most 20 halley iteration refinements of the value (usually finishes in 2)
    for _ in range(20):
        f = z0 - z * torch.exp(-z0)
        z01 = z0 + 1.0000001  # Numerical stability when z0 == -1
        delta = f / (z01 - (z0 + 2.) * f / (2. * z01))
        z0 = z0 - delta
        converged = torch.abs(delta) <= tolerance * torch.abs(z0)
        if torch.all(converged):
            break
    return z0


def _taylor_approx(z: torch.Tensor) -> torch.Tensor:
    """Compute an approximation of the lambertw function at z.

    Based on the polynomial expansion in https://arxiv.org/pdf/1003.1628.pdf. An empirical comparison of this polynomial
    expansion against the winitzki approximation found that this one is better when z < -0.2649.

    Args:
        z: The input to the lambertw function.

    Returns:
        An estimated value of lambertw(z).
    """
    p2 = 2 * (1. + math.e * z)
    p = torch.sqrt(p2)
    return -1. + p - p2 / 3. + 0.1527777777 * p2 * p


def _lambertw_winitzki_approx(z: torch.Tensor) -> torch.Tensor:
    """Compute an approximation of the lambertw function at z.

    Args:
        z: The input to the lambertw function.

    Returns:
        An estimated value of lambertw(z).
    """
    log1pz = torch.log1p(z)
    return log1pz * (1. - torch.log1p(log1pz) / (2. + log1pz))
