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
from typing import Any, Callable, Dict, List, Union, Optional

from fastestimator.search.search import Search


class GridSearch(Search):
    """A class which executes a grid search.

    Grid search can be used to evaluate or optimize results of one or more hyperparameters.

    ```python
    search = GridSearch(eval_fn=lambda search_idx, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
    search.fit()
    print(search.get_best_results()) # {'param': {'a': 6, 'b': 6, 'search_idx': 10}, 'result': {'value': 12}}

    search = GridSearch(eval_fn=lambda search_idx, a, b: {"sum": a + b}, params={"a": [1, 2, 3], "b": [4, 5, 6]})
    search.fit()
    print(search.get_best_results()) # {'param': {'a': 6, 'b': 6, 'search_idx': 10}, 'result': {'sum': 12}}
    ```

    Args:
        eval_fn: Function that evaluates result given parameter. One of its arguments must be 'search_idx' which will
            be automatically provided by the search routine. This can help with file saving / logging during the search.
            The eval_fn should return a dictionary, or else the return would be wrapped inside one.
        params: A dictionary with key names matching the `eval_fn`'s inputs. Its values should be lists of options.
        best_mode: Whether maximal or minimal objective is desired. Must be either 'min' or 'max'.
        optimize_field: the key corresponding to the target value when deciding the best. If None and multiple keys
            exist in result dictionary, the optimization is ambiguous therefore an error will be raised.
        name: The name of the search instance. This is used for saving and loading purposes.

    Raises:
        AssertionError: If `params` is not dictionary, or contains key not used by `eval_fn`
    """
    def __init__(self,
                 eval_fn: Callable[..., Union[Dict[str, Any], float]],
                 params: Dict[str, List],
                 best_mode: Optional[str] = None,
                 optimize_field: Optional[str] = None,
                 name: str = "grid_search"):
        assert isinstance(params, dict), "must provide params as a dictionary"

        eval_fn_args, params_args = set(inspect.signature(eval_fn).parameters.keys()), set(params.keys())
        assert eval_fn_args.issuperset(params_args), "unused param {} in eval_fn".format(params_args - eval_fn_args)
        super().__init__(eval_fn=eval_fn, name=name)
        self.params = params
        self.optimize_field = optimize_field
        self.best_mode = best_mode

    def _fit(self):
        experiments = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
        for exp in experiments:
            self.evaluate(**exp)
