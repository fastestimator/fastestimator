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
import itertools
from typing import Any, Callable, Dict, List

from fastestimator.search.search import Search


class GridSearch(Search):
    """A search class that performs the grid search.

    Grid search can be used to find the optimal combination of one or more hyperparameters based on its score function.

    ```python
    search = GridSearch(score_fn=lambda index, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
    search.fit()
    print(search.get_best_parameters()) # {"a": 3, "b": 6, "index": 9}
    ```

    Args:
        score_fn: Objective function that measures the fitness, index must be one of its argument.
        params: Dictionary with key name matching `score_fn`'s inputs, value to be the list of options.
        best_mode: Whether maximal or minimal fitness is desired, must be either 'min' or 'max'.
        name: The name of the search instance, this is used for saving and loading purpose.

    Raises:
        AssertionError: If `params` is not dictionary, or contains key not used by `score_fn`
    """
    def __init__(self,
                 score_fn: Callable[[Any], float],
                 params: Dict[str, List],
                 best_mode: str = "max",
                 name: str = "grid-search"):
        assert isinstance(params, dict), "must provide params as a dictionary"
        score_fn_args, params_args = set(inspect.signature(score_fn).parameters.keys()), set(params.keys())
        assert score_fn_args.issuperset(params_args), "unused param {} in score_fn".format(params_args - score_fn_args)
        super().__init__(score_fn=score_fn, best_mode=best_mode, name=name)
        self.params = params

    def _fit(self):
        experiments = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
        for exp in experiments:
            self.evaluate(**exp)
        print("FastEstimator-Search: grid search finished!")
