# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import Optional, Sequence, Union

from fastestimator.search.search import Search
from fastestimator.search.visualize.cartesian import visualize_cartesian, _cartesian_supports_data
from fastestimator.search.visualize.heatmap import visualize_heatmap, _heatmap_supports_data
from fastestimator.search.visualize.parallel_coordinate_plot import visualize_parallel_coordinates
from fastestimator.search.visualize.vis_util import SearchData, _load_search_file


def visualize_search(search: Union[Search, str],
                     title: Optional[str] = None,
                     ignore_keys: Union[None, str, Sequence[str]] = None,
                     save_path: Optional[str] = None,
                     verbose: bool = True,
                     **kwargs) -> None:
    """Visualize a given search instance, automatically choosing the most appropriate visualization technique to do so.

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        verbose: Whether to print out the save location.
        **kwargs: Arguments which can pass through the specific underlying visualizers, like 'groups' for cartesian
            plotting or 'color_by' for parallel coordinate plots.
    """
    if isinstance(search, str):
        search = _load_search_file(search)
    data = SearchData(search, ignore_keys=ignore_keys)
    if _cartesian_supports_data(data, throw_on_invalid=False):
        visualize_cartesian(search=search,
                            title=title,
                            ignore_keys=ignore_keys,
                            save_path=save_path,
                            verbose=verbose,
                            groups=kwargs.get("groups", None))
    elif _heatmap_supports_data(data, throw_on_invalid=False):
        visualize_heatmap(search=search, title=title, ignore_keys=ignore_keys, save_path=save_path, verbose=verbose)
    else:
        visualize_parallel_coordinates(search=search,
                                       title=title,
                                       ignore_keys=ignore_keys,
                                       save_path=save_path,
                                       verbose=verbose,
                                       color_by=kwargs.get('color_by', None))
