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
import inspect
import math
from typing import Any, Callable, Union

from fastestimator.search.search import Search


class GoldenSection(Search):
    """A search class that performs the golden-section search on single variable.

    Golden-section search is good at finding minimal or maximal value of a unimodal function, each time the search range
    is reduced by a constant factor - golden ratio. More details at https://en.wikipedia.org/wiki/Golden-section_search.

    ```python
    search = GoldenSection(score_fn=lambda index, n: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min")
    search.fit()
    print(search.get_best_parameters(index=False)) # {"n": 3}
    ```

    Args:
        score_fn: Objective function that measures the fitness, index must be one of its argument, the other argument is
            the variable to optimize.
        x_min: Lower limit (inclusive) of the search space.
        x_max: Upper limit (inclusive) of the search space.
        max_iter: Maximum number of iterations to run, the range at a given itertation i is 0.618**i * (x_max - x_min).
        integer: Whether the optimized variable is a discrete integer.
        best_mode: Whether maximal or minimal fitness is desired, must be either 'min' or 'max'.
        name: The name of the search instance, this is used for saving and loading purpose.

    Raises:
        AssertionError: If `score_fn`, `x_min`, `x_max` or `max_iter` arguments do not meet requirement.
    """
    def __init__(self,
                 score_fn: Callable[[Any], float],
                 x_min: Union[int, float],
                 x_max: Union[int, float],
                 max_iter: int,
                 integer: bool = True,
                 best_mode: str = "max",
                 name: str = "golden-section-search"):
        super().__init__(score_fn=score_fn, best_mode=best_mode, name=name)
        assert x_min < x_max, "x_min must be smaller than x_max"
        if integer:
            assert isinstance(x_min, int) and isinstance(x_max, int), "x_min and x_max must be integer"
        args = set(inspect.signature(score_fn).parameters.keys()) - {'index'}
        assert len(args) == 1, "the score function should only contain one argument other than 'index'"
        assert max_iter > 2, "max_iter should be greater than 2"
        self.x_min = x_min
        self.x_max = x_max
        self.max_iter = max_iter
        self.integer = integer
        self.arg_name = args.pop()

    def _convert(self, x: Union[float, int]):
        return int(x) if self.integer else x

    def _is_better(self, a: float, b: float):
        return a > b if self.best_mode == "max" else a < b

    def _fit(self):
        a, b, h = self.x_min, self.x_max, self.x_max - self.x_min
        invphi, invphi2 = (math.sqrt(5) - 1) / 2, (3 - math.sqrt(5)) / 2
        c, d = self._convert(a + invphi2 * h), self._convert(a + invphi * h)
        yc = self.evaluate(**{self.arg_name: c})
        yd = self.evaluate(**{self.arg_name: d})
        for _ in range(self.max_iter - 2):
            if self._is_better(yc, yd):
                b, d, yd = d, c, yc
                h = invphi * h
                c = self._convert(a + invphi2 * h)
                yc = self.evaluate(**{self.arg_name: c})
            else:
                a, c, yc = c, d, yd
                h = invphi * h
                d = self._convert(a + invphi * h)
                yd = self.evaluate(**{self.arg_name: d})
        print("FastEstimator-Search: golden section search finished!")
