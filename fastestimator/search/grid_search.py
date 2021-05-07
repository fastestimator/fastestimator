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
from typing import Callable, Dict, List

from fastestimator.search.search import Search


class GridSearch(Search):
    """A class which executes a grid search.

    Grid search can be used to find the optimal combination of one or more hyperparameters.

    ```python
    search = GridSearch(score_fn=lambda search_idx, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
    search.fit()
    print(search.get_best_parameters()) # {"a": 3, "b": 6, "search_idx": 9}
    ```

    Args:
        score_fn: Objective function that measures search fitness. One of its arguments must be 'search_idx' which will
            be automatically provided by the search routine. This can help with file saving / logging during the search.
        params: A dictionary with key names matching the `score_fn`'s inputs. Its values should be lists of options.
        best_mode: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.
        name: The name of the search instance. This is used for saving and loading purposes.

    Raises:
        AssertionError: If `params` is not dictionary, or contains key not used by `score_fn`
    """
    def __init__(self,
                 score_fn: Callable[..., float],
                 params: Dict[str, List],
                 best_mode: str = "max",
                 name: str = "grid_search"):
        assert isinstance(params, dict), "must provide params as a dictionary"
        score_fn_args, params_args = set(inspect.signature(score_fn).parameters.keys()), set(params.keys())
        assert score_fn_args.issuperset(params_args), "unused param {} in score_fn".format(params_args - score_fn_args)
        super().__init__(score_fn=score_fn, best_mode=best_mode, name=name)
        self.params = params

    def _fit(self):
        experiments = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
        for exp in experiments:
            self.evaluate(**exp)
        best_results = self.get_best_results()
        print("FastEstimator-Search: Grid Search Finished, best parameters: {}, best score: {}".format(
            best_results[0], best_results[1]))
