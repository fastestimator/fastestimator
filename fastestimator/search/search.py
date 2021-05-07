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
from abc import ABC, abstractmethod
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Tuple

from fastestimator.network import build

class Search(ABC):
    """Base class which other searches inherit from.

    The base search class takes care of evaluation logging, saving and loading, and is also able to recover from
    interrupted search runs and cache the search history.

    Args:
        score_fn: Objective function that measures search fitness. One of its arguments must be 'search_idx' which will
            be automatically provided by the search routine. This can help with file saving / logging during the search.
        best_mode: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.
        name: The name of the search instance. This is used for saving and loading purposes.

    Raises:
        AssertionError: If `best_mode` is not 'min' or 'max', or search_idx is not an input argument of `score_fn`.
    """
    def __init__(self, score_fn: Callable[..., float], best_mode: str = "max", name: str = "search"):
        assert best_mode in ["max", "min"], "best_mode must be either 'max' or 'min'"
        assert "search_idx" in inspect.signature(score_fn).parameters, \
            "score_fn must take 'search_idx' as one of its input arguments"
        self.score_fn = score_fn
        self.best_mode = best_mode
        self.name = name
        self.save_dir = None
        self._initialize_state()

    def _initialize_state(self):
        self.search_idx = 0
        self.search_results = []
        self.evaluation_cache = {}

    def evaluate(self, **kwargs: Any) -> float:
        """Evaluate the score function and return the score.

        Args:
            kwargs: Any keyword argument(s) to pass to the score function. Should not contain search_idx as this will be
                populated manually here.

        Returns:
            Fitness score calculated by `score_fn`.
        """
        # evaluation caching
        hash_value = hash(tuple(sorted(kwargs.items())))
        if hash_value in self.evaluation_cache:
            score = self.evaluation_cache[hash_value]
        else:
            self.search_idx += 1
            build.count = 0  # Resetting the build count to refresh the model names
            kwargs["search_idx"] = self.search_idx
            score = self.score_fn(**kwargs)
            self.search_results.append((kwargs, score))
            self.evaluation_cache[hash_value] = score
            if self.save_dir is not None:
                self.save(self.save_dir)
            print("FastEstimator-Search: Evaluated {}, score: {}".format(kwargs, score))
        return score

    def get_best_results(self) -> Tuple[Dict[str, Any], float]:
        """Get the best result from the current search history.

        Returns:
            The best results in the format of (parameter, score)

        Raises:
            RuntimeError: If the search hasn't been run yet.
        """
        if not self.search_results:
            raise RuntimeError("No search has been run yet, so best parameters are not available.")
        if self.best_mode == "max":
            best_results = max(self.search_results, key=lambda x: x[1])
        else:  # min
            best_results = min(self.search_results, key=lambda x: x[1])
        return best_results

    def get_search_results(self) -> List[Tuple[Dict[str, Any], float]]:
        """Get the current search history.

        Returns:
            The evluation history list, with each element being a tuple of parameters and score.
        """
        return self.search_results.copy()

    def _get_state(self) -> Dict[str, Any]:
        """Get the current state of the search instance, the state is the variables that can be saved or loaded.

        Returns:
            The dictionary containing the state variable.
        """
        return {"search_results": self.search_results}

    def save(self, save_dir: str) -> None:
        """Save the state of the instance to a specific directory, it will create `name.json` file in the `save_dir`.

        Args:
            save_dir: The folder path to save to.
        """
        file_path = os.path.join(save_dir, "{}.json".format(self.name))
        with open(file_path, 'w') as fp:
            json.dump(self._get_state(), fp, indent=4)
        print("FastEstimator-Search: Saving the search state to {}".format(file_path))

    def load(self, load_dir: str, not_exist_ok: bool = False) -> None:
        """Load the state of search from a given directory. It will look for `name.json` within the `load_dir`.

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
            # restore all state variables from get_state
            self.__dict__.update(state)
            # restore evaluation cache and search_idx
            for kwargs, score in self.search_results:
                kwargs = kwargs.copy()
                search_idx = kwargs.pop('search_idx')  # This won't appear in the hash later
                self.search_idx = self.search_idx if self.search_idx > search_idx else search_idx
                # Each python session uses a unique salt for hash, so can't save the hashes to disk for re-use
                self.evaluation_cache[hash(tuple(sorted(kwargs.items())))] = score
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
            self.load(save_dir, not_exist_ok=True)
        self._fit()

    @abstractmethod
    def _fit(self) -> None:
        raise NotImplementedError
