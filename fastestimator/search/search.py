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
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import fastestimator as fe


class Search:
    """Base class which other searches inherit from.

    The base search class takes care of evaluation logging, saving and loading, and is also able to recover from
    interrupted search runs and cache the search history.

    Args:
        eval_fn: Function that evaluates result given parameter. One of its arguments must be 'search_idx' which will
            be automatically provided by the search routine. This can help with file saving / logging during the search.
            The eval_fn should return a dictionary, or else the return would be wrapped inside one.
        name: The name of the search instance. This is used for saving and loading purposes.

    Raises:
        AssertionError: If `best_mode` is not 'min' or 'max', or search_idx is not an input argument of `eval_fn`.
    """
    def __init__(self, eval_fn: Callable[..., Dict], name: str = "search"):
        assert "search_idx" in inspect.signature(eval_fn).parameters, \
            "eval_fn must take 'search_idx' as one of its input arguments"
        self.eval_fn = eval_fn
        self.name = name
        self.save_dir = None
        self.best_mode = None
        self.optimize_field = None
        self._initialize_state()

    def _initialize_state(self):
        self.search_idx = 0
        self.search_summary = []
        self.evaluation_cache = {}

    def evaluate(self, **kwargs: Any) -> Dict[str, Union[float, int, str]]:
        """Evaluate the eval_fn and return the result.

        Args:
            kwargs: Any keyword argument(s) to pass to the score function. Should not contain search_idx as this will be
                populated manually here.

        Returns:
            result returned by `eval_fn`.
        """
        # evaluation caching
        hash_value = hash(tuple(sorted(kwargs.items())))
        if hash_value in self.evaluation_cache:
            result = self.evaluation_cache[hash_value]
        else:
            self.search_idx += 1
            fe.fe_build_count = 0  # Resetting the build count to refresh the model names
            kwargs["search_idx"] = self.search_idx
            result = self.eval_fn(**kwargs)
            if not isinstance(result, dict):
                result = {"value": result}
            summary = {"param": kwargs, "result": result}
            self.search_summary.append(summary)
            self.evaluation_cache[hash_value] = result
            if self.save_dir is not None:
                self.save(self.save_dir)
            print("FastEstimator-Search: Evaluated {}, result: {}".format(kwargs, result))
        return result

    def _infer_optimize_field(self, result: Dict[str, Any]) -> str:
        """Infer optimize_field based on result, only needed when optimize_field is not provided.

        Returns:
            The optimize_field.

        Raises:
            Value error if multiple keys exist in the result.
        """
        if len(self.search_summary[0]['result']) == 1:
            optimize_field = list(result.keys())[0]
        else:
            raise ValueError("Multiple keys exist in result dictionary and optimize_field is None.")
        return optimize_field

    def get_best_results(self,
                         best_mode: Optional[str] = None,
                         optimize_field: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get the best result from the current search summary.

        Args:
            best_mode: Whether maximal or minimal objective is desired. Must be either 'min' or 'max'.
            optimize_field: the key corresponding to the target value when deciding the best. If None and multiple keys
                exist in result dictionary, the optimization is ambiguous therefore an error will be raised.

        Returns:
            The best results in the format of {"param":parameter, "result": result}
        """
        optimize_field = optimize_field or self.optimize_field
        best_mode = best_mode or self.best_mode
        assert best_mode in ["max", "min"], "best_mode must be either 'max' or 'min'"
        if not self.search_summary:
            raise RuntimeError("No search summary yet, so best parameters are not available.")
        if optimize_field is None:
            optimize_field = self._infer_optimize_field(self.search_summary[0]['result'])
        if best_mode == "max":
            best_results = max(self.search_summary, key=lambda x: x['result'][optimize_field])
        else:  # min
            best_results = min(self.search_summary, key=lambda x: x['result'][optimize_field])
        return best_results

    def _get_state(self) -> Dict[str, Any]:
        """Get the current state of the search instance, the state is the variables that can be saved or loaded.

        Returns:
            The dictionary containing the state variable.
        """
        state = {"search_summary": self.search_summary}
        # Include extra info if it is available for better visualization options later
        if self.name:
            state["name"] = self.name
        if self.best_mode:
            state["best_mode"] = self.best_mode
        if self.optimize_field:
            state["optimize_field"] = self.optimize_field
        return state

    def get_search_summary(self) -> List[Dict[str, Dict[str, Any]]]:
        """Get the current search history.

        Returns:
            The evaluation history list, with each element being a tuple of parameters and score.
        """
        return self.search_summary.copy()

    def save(self, save_dir: str) -> None:
        """Save the state of the instance to a specific directory, it will create `name.json` file in the `save_dir`.

        Args:
            save_dir: The folder path to save to.
        """
        file_path = os.path.join(save_dir, "{}.json".format(self.name))
        with open(file_path, 'w') as fp:
            json.dump(self._get_state(), fp, indent=4)
        print("FastEstimator-Search: Saving the search summary to {}".format(file_path))

    def load(self, load_dir: str, not_exist_ok: bool = False) -> None:
        """Load the search summary from a given directory. It will look for `name.json` within the `load_dir`.

        Args:
            load_dir: The folder path to load the state from.
            not_exist_ok: whether to ignore when the file does not exist.
        """
        self._initialize_state()
        # load from file
        load_dir = os.path.abspath(os.path.normpath(load_dir))
        file_path = os.path.join(load_dir, "{}.json".format(self.name))
        if os.path.exists(file_path):
            with open(file_path, 'r') as fp:
                state = json.load(fp)
            self.__dict__.update(state)
            # restore evaluation cache and search_idx
            for summary in self.search_summary:
                kwargs = summary['param'].copy()
                search_idx = kwargs.pop('search_idx')  # This won't appear in the hash later
                self.search_idx = self.search_idx if self.search_idx > search_idx else search_idx
                # Each python session uses a unique salt for hash, so can't save the hashes to disk for re-use
                self._make_hashable(kwargs)
                self.evaluation_cache[hash(tuple(sorted(kwargs.items())))] = summary['result']
            print("FastEstimator-Search: Loading the search state from {}".format(file_path))
        elif not not_exist_ok:
            raise ValueError("cannot find file to load in {}".format(file_path))

    def fit(self, save_dir: str = None) -> None:
        """Start the search.

        Args:
            save_dir: When `save_dir` is provided, the search results will be backed up to the `save_dir` after each
                evaluation. It will also attempt to load the search state from `save_dir` if possible. This is useful
                when the search might experience interruption since it can be restarted using the same command.
        """
        if save_dir is None:
            self._initialize_state()
        else:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.load(save_dir, not_exist_ok=True)
        self._fit()

    def _fit(self) -> None:
        raise NotImplementedError

    def _make_hashable(self, kwargs):
        for key in kwargs:
            if isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
