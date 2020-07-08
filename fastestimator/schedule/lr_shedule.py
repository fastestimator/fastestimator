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
import math


def cosine_decay(time: int,
                 cycle_length: int,
                 init_lr: float,
                 min_lr: float = 1e-6,
                 start: int = 1,
                 cycle_multiplier: int = 1):
    """Learning rate cosine decay function (using half of cosine curve).

    This method is useful for scheduling learning rates which oscillate over time:
    ```python
    s = fe.schedule.LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])
    ```

    For more information, check out SGDR: https://arxiv.org/pdf/1608.03983.pdf.

    Args:
        time: The current step or epoch during training starting from 1.
        cycle_length: The decay cycle length.
        init_lr: Initial learning rate to decay from.
        min_lr: Minimum learning rate.
        start: The step or epoch to start the decay schedule.
        cycle_multiplier: The factor by which next cycle length will be multiplied.

    Returns:
        lr: learning rate given current step or epoch.
    """
    if time < start:
        lr = init_lr
    else:
        time = time - start + 1
        if cycle_multiplier > 1:
            current_cycle_idx = math.ceil(
                math.log(time * (cycle_multiplier - 1) / cycle_length + 1) / math.log(cycle_multiplier)) - 1
            cumulative = cycle_length * (cycle_multiplier**current_cycle_idx - 1) / (cycle_multiplier - 1)
        elif cycle_multiplier == 1:
            current_cycle_idx = math.ceil(time / cycle_length) - 1
            cumulative = current_cycle_idx * cycle_length
        else:
            raise ValueError("multiplier must be at least 1")
        current_cycle_length = cycle_length * cycle_multiplier**current_cycle_idx
        time_in_cycle = (time - cumulative) / current_cycle_length
        lr = (init_lr - min_lr) / 2 * math.cos(time_in_cycle * math.pi) + (init_lr + min_lr) / 2
    return lr
