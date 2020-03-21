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
from typing import Iterable, Optional, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.util import NonContext

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def get_gradient(target: Tensor,
                 sources: Union[Iterable[Tensor], Tensor],
                 higher_order: bool = False,
                 tape: Optional[tf.GradientTape] = None,
                 retain_graph: bool = True) -> Union[Iterable[Tensor], Tensor]:
    """calculate gradients of target w.r.t sources

    Args:
        target: target tensor
        sources : sequence of source tensors
        higher_order : whether the gradient will be used for higher order gradients. Defaults to False.
        tape : tensorflow gradient tape, only needed when using tensorflow backend. Defaults to None.
        retain_graph : whether to retain pytorch graph, only valid under pytorch backend. Defaults to True.

    Returns:
        gradients as sequence of tensors
    """
    if tape:
        with NonContext() if higher_order else tape.stop_recording():
            gradients = tape.gradient(target, sources)
    else:
        gradients = torch.autograd.grad(target,
                                        sources,
                                        retain_graph=retain_graph,
                                        create_graph=higher_order,
                                        only_inputs=True)
    return gradients
