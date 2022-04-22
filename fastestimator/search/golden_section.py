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
from typing import Any, Callable, Dict, Optional, Union

from fastestimator.search.search import Search


class GoldenSection(Search):
    """A search class that performs the golden-section search on a single variable.

    Golden-section search is good at finding minimal or maximal values of a unimodal function. Each search step reduces
    the search range by a constant factor: the inverse of golden ratio. More details are available at:
    https://en.wikipedia.org/wiki/Golden-section_search.

    ```python
    search = GoldenSection(eval_fn=lambda search_idx, n: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min")
    search.fit()
    print(search.get_best_parameters()) # {'param': {'n': 3, 'search_idx': 2}, 'result': {'value': 0}}
    ```

    Args:
        eval_fn: Function that evaluates result given parameter. One of its arguments must be 'search_idx' which will
            be automatically provided by the search routine. This can help with file saving / logging during the search.
            The eval_fn should return a dictionary, or else the return would be wrapped inside one. The other
            argument should be the variable to be searched over.
        x_min: Lower limit (inclusive) of the search space.
        x_max: Upper limit (inclusive) of the search space.
        max_iter: Maximum number of iterations to run. The range at a given iteration i is 0.618**i * (x_max - x_min).
            Note that the eval_fn will always be evaluated twice before any iterations begin.
        best_mode: Whether maximal or minimal objective is desired. Must be either 'min' or 'max'.
        optimize_field: the key corresponding to the target value when deciding the best. If None and multiple keys
            exist in result dictionary, the optimization is ambiguous therefore an error will be raised.
        integer: Whether the optimized variable is a discrete integer.
        best_mode: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.
        name: The name of the search instance. This is used for saving and loading purposes.

    Raises:
        AssertionError: If `eval_fn`, `x_min`, `x_max`, or `max_iter` are invalid.
    """
    def __init__(self,
                 eval_fn: Callable[[int, Union[int, float]], float],
                 x_min: Union[int, float],
                 x_max: Union[int, float],
                 max_iter: int,
                 best_mode: str,
                 optimize_field: Optional[str] = None,
                 integer: bool = True,
                 name: str = "golden_section_search"):
        super().__init__(eval_fn=eval_fn, name=name)
        assert best_mode in ["max", "min"], "best_mode must be either 'max' or 'min'"
        assert x_min < x_max, "x_min must be smaller than x_max"
        if integer:
            assert isinstance(x_min, int) and isinstance(x_max, int), \
                "x_min and x_max must be integers when searching in integer mode"
        args = set(inspect.signature(eval_fn).parameters.keys()) - {'search_idx'}
        assert len(args) == 1, "the score function should only contain one argument other than 'search_idx'"
        assert max_iter > 0, "max_iter should be greater than 0"
        self.x_min = x_min
        self.x_max = x_max
        self.max_iter = max_iter
        self.best_mode = best_mode
        self.optimize_field = optimize_field
        self.integer = integer
        self.arg_name = args.pop()

    def _convert(self, x: Union[float, int]) -> Union[int, float]:
        return int(x) if self.integer else x

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.best_mode == "max" else a < b

    def _fit(self):
        a, b, h = self.x_min, self.x_max, self.x_max - self.x_min
        invphi, invphi2 = (math.sqrt(5) - 1) / 2, (3 - math.sqrt(5)) / 2
        c, d = self._convert(a + invphi2 * h), self._convert(a + invphi * h)
        yc = self._get_value_from_result(self.evaluate(**{self.arg_name: c}))
        yd = self._get_value_from_result(self.evaluate(**{self.arg_name: d}))
        for _ in range(self.max_iter):
            if self._is_better(yc, yd):
                b, d, yd = d, c, yc
                h = invphi * h
                c = self._convert(a + invphi2 * h)
                yc = self._get_value_from_result(self.evaluate(**{self.arg_name: c}))
            else:
                a, c, yc = c, d, yd
                h = invphi * h
                d = self._convert(a + invphi * h)
                yd = self._get_value_from_result(self.evaluate(**{self.arg_name: d}))
        best_results = self.get_best_results()
        print("FastEstimator-Search: Golden Section Search Finished, best parameters: {}, best result: {}".format(
            best_results['param'], best_results['result']))

    def _get_value_from_result(self, result: Dict[str, Any]) -> Union[int, float]:
        optimize_field = self.optimize_field
        if optimize_field is None:
            optimize_field = self._infer_optimize_field(result)
        return result[optimize_field]
