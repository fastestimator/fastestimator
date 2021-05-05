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
from typing import Any, Callable, Dict, List, Tuple


class Search:
    """Base search class that other search classes inherit from.

    The base search class takes care of the evaluation logging, saving and loading, it is also able to recover from
    interrupted search runs and cache the search history.

    Args:
        score_fn: Objective function that measures the fitness, index must be one of its argument.
        best_mode: Whether maximal or minimal fitness is desired, must be either 'min' or 'max'.
        name: The name of the search instance, this is used for saving and loading purpose.

    Raises:
        AssertionError: If `best_mode` is not 'min' or 'max', or index is not an input argument of `score_fn`.
    """
    def __init__(self, score_fn: Callable[[Any], float], best_mode: str = "max", name: str = "search"):
        assert best_mode in ["max", "min"], "best_mode must be either 'max' or 'min'"
        assert "index" in inspect.signature(score_fn).parameters, "score_fn must take 'index' as one of its input arg"
        self.score_fn = score_fn
        self.best_mode = best_mode
        self.name = name
        self.save_dir = None
        self._initialize_state()

    def _initialize_state(self):
        self.index = 0
        self.search_result = []
        self.evaluation_cache = {}

    def evaluate(self, **kwargs: Any) -> float:
        """Evaluate the score function and return the score.

        Args:
            kwargs: Any keyword argument.

        Returns:
            Fitness score calculated by `score_fn`.
        """
        # evaluation caching
        if hash(tuple(sorted(kwargs.items()))) in self.evaluation_cache:
            score = self.evaluation_cache[hash(tuple(sorted(kwargs.items())))]
        else:
            self.index += 1
            hash_value = hash(tuple(sorted(kwargs.items())))
            kwargs["index"] = self.index
            score = self.score_fn(**kwargs)
            self.search_result.append((kwargs, score))
            self.evaluation_cache[hash_value] = score
            if self.save_dir is not None:
                self.save(self.save_dir)
        return score

    def get_best_parameters(self, display_index: bool = True) -> Dict[str, Any]:
        """Get the best parameter from the current search history.

        Args:
            display_index: Whether to display experiment index in the final parameter output.

        Returns:
            The parameter (in dictioanry) that corresponds to the best score.
        """
        if self.best_mode == "max":
            best_params = max(self.search_result, key=lambda x: x[1])[0]
        elif self.best_mode == "min":
            best_params = min(self.search_result, key=lambda x: x[1])[0]
        if not display_index:
            best_params.pop('index', None)
        return best_params

    def get_search_results(self) -> List[Tuple[Dict[str, Any], float]]:
        """Get the current search history.

        Returns:
            The evluation history list, with each element to be a tuple of parameter and score.
        """
        return self.search_result

    def get_state(self) -> Dict[Any, Any]:
        """Get the current state of the search instance, the state is the variables that can be saved or loaded.

        Returns:
            The dictionary containing the state variable.
        """
        return {"index": self.index, "search_result": self.search_result}

    def save(self, save_dir: str):
        """Save the state of the instance to a specific directory, it will create `name.json` file in the `save_dir`.

        Args:
            save_dir: The folder path to save to.
        """
        file_path = os.path.join(save_dir, "{}.json".format(self.name))
        with open(file_path, 'w') as fp:
            json.dump(self.get_state(), fp, indent=4)
        print("FastEstimator-Search: Saving the search state to {}".format(file_path))

    def load(self, load_dir: str, not_exist_ok: bool = False):
        """Load the state of search from a given directory, it will look for `name.json` from the `load_dir`.

        Args:
            load_dir: The folder path to load the state from.
            not_exist_ok: whether to ignore when the file does not exist.
        """
        self._initialize_state()
        # load from file
        file_path = os.path.join(load_dir, "{}.json".format(self.name))
        if os.path.exists(file_path):
            with open(file_path, 'r') as fp:
                state = json.load(fp)
            # restore all state variables from get_state
            for key, value in state.items():
                exec("self.{} = value".format(key))
            # restore evaluation cache (not a state variable)
            for kwarg, score in self.search_result:
                kwarg_no_index = {key: value for key, value in kwarg.items() if key != "index"}
                self.evaluation_cache[hash(tuple(sorted(kwarg_no_index.items())))] = score
            print("FastEstimator-Search: Loading the search state from {}".format(file_path))
        elif not not_exist_ok:
            raise ValueError("cannot find file to load in {}".format(file_path))

    def fit(self, save_dir: str = None):
        """Start the search.

        Args:
            save_dir: When `save_dir` is provided, the search results will be backed up to the `save_dir` after each
                evaluation, in addition, it will load the search state from `save_dir` if possible. This is useful when
                the search might experience interruption, using the same command can allow for self-recovery.
        """
        if save_dir is None:
            self._initialize_state()
        else:
            self.save_dir = save_dir
            self.load(save_dir, not_exist_ok=True)
        self._fit()

    def _fit(self):
        raise NotImplementedError
