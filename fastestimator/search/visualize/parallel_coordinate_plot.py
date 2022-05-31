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

from natsort import humansorted
from plotly.graph_objects import Parcoords

from fastestimator.search.search import Search
from fastestimator.search.visualize.vis_util import SearchData, _load_search_file
from fastestimator.util.base_util import in_notebook, FigureFE


def plot_parallel_coordinates(search: Union[Search, str],
                              color_by: Optional[str] = None,
                              title: Optional[str] = None,
                              ignore_keys: Union[None, str, Sequence[str]] = None) -> FigureFE:
    """Draw a parallel coordinate plot based on search results.

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        color_by: Which key to use for line coloring.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.

    Returns:
        A plotly figure instance.
    """
    if isinstance(search, str):
        search = _load_search_file(search)
    if color_by is None:
        color_by = search.optimize_field
    if title is None:
        title = search.name
    reverse_colors = search.best_mode == 'min'
    search = SearchData(search=search, ignore_keys=ignore_keys)
    if not search.data:
        return FigureFE()

    # Finalize the result column to color by if none has been inferred yet
    if color_by is None:
        candidates = set(search.results) - set(search.categorical_maps.keys())
        if candidates:
            # Prefer numeric columns first
            color_by = humansorted(candidates)[-1]
        else:
            # Fall back to categorical columns
            color_by = humansorted(search.results)[-1]

    # Currently can't edit line width, but hopefully supported in the future:
    # https://github.com/plotly/plotly.js/issues/2573

    fig = Parcoords(line={'colorscale': 'Viridis',
                          'color': search.data[color_by],
                          'colorbar': {'title': color_by},
                          'showscale': True,
                          'reversescale': reverse_colors},
                    dimensions=[{'label': x,
                                 'values': search.data[x],
                                 'tickvals': list(
                                     search.categorical_maps[x].values()) if x in search.categorical_maps else None,
                                 'ticktext': list(
                                     search.categorical_maps[x].keys()) if x in search.categorical_maps else None,
                                 } for x in search.params + search.results],
                    labelfont={'size': 12},
                    tickfont={'size': 11},
                    rangefont={'size': 12})

    fig = FigureFE(data=fig, layout={'title': title, 'title_x': 0.5})

    # If inside a jupyter notebook then force the height larger
    if in_notebook():
        fig.update_layout(height=500)

    return fig


def visualize_parallel_coordinates(search: Union[Search, str],
                                   color_by: Optional[str] = None,
                                   title: Optional[str] = None,
                                   ignore_keys: Union[None, str, Sequence[str]] = None,
                                   save_path: Optional[str] = None,
                                   verbose: bool = True) -> None:
    """Display or save a parallel coordinate plot based on search results.

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        color_by: Which key to use for line coloring.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        verbose: Whether to print out the save location.
    """
    fig = plot_parallel_coordinates(search=search, color_by=color_by, title=title, ignore_keys=ignore_keys)
    fig.show(save_path=save_path, verbose=verbose, scale=2)
