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
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Union

from fastestimator.util.traceability_util import FeSummaryTable


class ValWithError(NamedTuple):
    """A class to record values with error bars (for special visualization in the logger).
    """
    y_min: float
    y: float
    y_max: float

    def __str__(self):
        return f"({self.y_min}, {self.y}, {self.y_max})"


class Summary:
    """A summary object that records training history.

    This class is intentionally not @traceable.

    Args:
        name: Name of the experiment. If None then experiment results will be ignored.
        system_config: A description of the initialization parameters defining the estimator associated with this
            experiment.
    """
    def __init__(self, name: Optional[str], system_config: Optional[List[FeSummaryTable]] = None) -> None:
        self.name = name
        self.system_config = system_config
        self.history = defaultdict(lambda: defaultdict(dict))  # {mode: {key: {step: value}}}

    def merge(self, other: 'Summary'):
        """Merge another `Summary` into this one.

        Args:
            other: Other `summary` object to be merged.
        """
        for mode, sub in other.history.items():
            for key, val in sub.items():
                self.history[mode][key].update(val)

    def __bool__(self) -> bool:
        """Whether training history should be recorded.

        Returns:
            True iff this `Summary` has a non-None name.
        """
        return bool(self.name)

    def __getstate__(self) -> Dict[str, Any]:
        """Get a representation of the state of this object.

        This method is invoked by pickle.

        Returns:
            The information to be recorded by a pickle summary of this object.
        """
        state = self.__dict__.copy()
        del state['system_config']
        state['history'] = dict(state['history'])
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set this objects internal state from a dictionary of variables.

        This method is invoked by pickle.

        Args:
            state: The saved state to be used by this object.
        """
        history = defaultdict(lambda: defaultdict(dict))
        history.update(state.get('history', {}))
        state['history'] = history
        self.__dict__.update(state)


def average_summaries(name: str, summaries: List[Summary]) -> Summary:
    """Average multiple summaries together, storing their metric means +- stdevs.

    Args:
        name: The name for the new summary.
        summaries: A list of summaries to be averaged.

    Returns:
        A single summary object reporting mean+-stddev for each metric. If a particular value has only 1 datapoint, it
        will not be averaged.
    """
    if len(summaries) == 0:
        return Summary(name=name)
    if len(summaries) == 1:
        summaries[0].name = name
        return summaries[0]
    consolidated = Summary(name=name)
    # Find all of the modes, keys, and steps over the various summaries
    modes = {mode for summary in summaries for mode in summary.history.keys()}
    keys = {key for summary in summaries for key_pairs in summary.history.values() for key in key_pairs.keys()}
    steps = {
        step
        for summary in summaries for key_pairs in summary.history.values() for val_pair in key_pairs.values()
        for step in val_pair.keys()
    }
    # Average everything
    for mode in modes:
        for key in keys:
            if key == 'epoch':
                # doesn't make sense to average the epoch over different summaries
                # TODO - if all summaries have same epoch dict then preserve it
                continue
            for step in steps:
                vals = []
                for summary in summaries:
                    history = summary.history
                    if mode in history and key in history[mode] and step in history[mode][key]:
                        val = history[mode][key][step]
                        if isinstance(val, str):
                            # Can't plot strings over time...
                            val = [float(s) for s in re.findall(r'(\d+\.\d+|\.?\d+)', val)]
                            if len(val) == 1:
                                # We got an unambiguous number
                                val = val[0]
                            else:
                                val = None
                        elif isinstance(val, ValWithError):
                            val = val.y
                        elif not isinstance(val, (int, float)):
                            val = None
                        if val is not None:
                            vals.append(val)
                if mode == 'test':
                    # We will consolidate these later
                    val = vals
                else:
                    val = _reduce_list(vals)
                if val is None:
                    continue
                consolidated.history[mode][key][step] = val
        if mode == 'test':
            # Due to early stopping, the test mode might be invoked at different steps/epochs. These values will be
            # merged and assigned to the largest available step.
            for key, step_val in consolidated.history[mode].items():
                vals = []
                for step, val in step_val.items():
                    if isinstance(val, list):
                        vals.extend(val)
                    else:
                        vals.append(val)
                step = max(step_val.keys())
                val = _reduce_list(vals)
                consolidated.history[mode][key].clear()
                if val is not None:
                    consolidated.history[mode][key][step] = val
    return consolidated


def _reduce_list(vals: List[Union[int, float]]) -> Union[None, int, float, ValWithError]:
    """Convert a list of numbers into a consolidated summary of those numbers.

    Args:
        vals: A list of values to be summarized.

    Returns:
        None if `vals` is empty, otherwise (mean - std, mean, mean + std) if multiple non-equal values are provided, or
        else simply the value if only a single (possibly repeated) value is found.
    """
    val = None
    if len(vals) > 1:
        if vals[0] == vals[1] and vals.count(vals[0]) == len(vals):
            # If all values are exactly the same, then return single value rather than computing std. The first check is
            # to give fast short-circuiting
            val = vals[0]
        else:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            val = ValWithError(mean - std, mean, mean + std)
    elif len(vals) == 1:
        val = vals[0]
    return val
