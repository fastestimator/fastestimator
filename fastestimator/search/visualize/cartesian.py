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
import math
from collections import defaultdict
from typing import List, Optional, Sequence, Union

from natsort import humansorted
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots

from fastestimator.search.search import Search
from fastestimator.search.visualize.vis_util import SearchData, _load_search_file
from fastestimator.util.base_util import FigureFE, get_colors, in_notebook


def _cartesian_supports_data(data: SearchData, throw_on_invalid: bool = True) -> bool:
    """Determine whether cartesian visualization supports the given data.

    Args:
        data: The data to be visualized.
        throw_on_invalid: Whether to throw an exception if the data is invalid.

    Returns:
        True iff the data can be visualized via cartesian plots.
    """
    if len(data.params) != 1:
        if not throw_on_invalid:
            return False
        raise ValueError(f"Cartesian plots require exactly 1 param, but found {len(data.params)}: {data.params}")
    if len(data.results) == 0:
        if not throw_on_invalid:
            return False
        raise ValueError("Cartesian plots require at least 1 result, but none were found")
    if data.categorical_maps.keys():
        if not throw_on_invalid:
            return False
        raise ValueError(f"Cartesian plots only support numerical values, but the following categorical features were "
                         f"found: {','.join(data.categorical_maps.keys())}")
    return True


def plot_cartesian(search: Union[Search, str],
                   title: Optional[str] = None,
                   ignore_keys: Union[None, str, Sequence[str]] = None,
                   groups: Optional[List[List[str]]] = None) -> FigureFE:
    """Draw cartesian plot(s) based on search results.

    Requires exactly 1 param and 1+ results (after accounting for the ignore_keys).

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.
        groups: Which result keys should be plotted on the same y-axis, rather than on separate subplots.

    Returns:
        A plotly figure instance.
    """
    if isinstance(search, str):
        search = _load_search_file(search)
    if title is None:
        title = search.name
    search = SearchData(search=search, ignore_keys=ignore_keys)
    _cartesian_supports_data(search)

    # Figure out how the plots will be grouped together
    if groups is None:
        groups = [[x] for x in search.results]
    leftovers = {x for x in search.results}
    for group in groups:
        for elem in group:
            leftovers.discard(elem)
            if elem not in search.results:
                raise ValueError(f"The key '{elem}' was specified in 'groups', but was not found in the search "
                                 f"results. Available keys are {search.results}")
    groups = groups + [[x] for x in humansorted(leftovers)]

    # Map the metrics into an n x n grid, then remove any extra columns. Final grid will be n x m with m <= n
    n_plots = len(groups)
    n_rows = math.ceil(math.sqrt(n_plots))
    n_cols = math.ceil(n_plots / n_rows)

    # Get basic plot layout
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes='all')
    fig.update_layout({
        'plot_bgcolor': '#FFF',
        'hovermode': 'closest',
        'margin': {
            't': 50
        },
        'modebar': {
            'add': ['hoverclosest', 'hovercompare'], 'remove': ['select2d', 'lasso2d']
        },
        'legend': {
            'tracegroupgap': 5, 'font': {
                'size': 11
            }
        },
        'title': title,
        'title_x': 0.5
    })

    # Configure x and y labels
    for idx, group in enumerate(groups, start=1):
        plotly_idx = idx if idx > 1 else ""
        x_axis_name = f'xaxis{plotly_idx}'
        y_axis_name = f'yaxis{plotly_idx}'
        fig['layout'][x_axis_name]['title'] = search.params[0]
        fig['layout'][x_axis_name]['showticklabels'] = True
        fig['layout'][x_axis_name]['linecolor'] = "#BCCCDC"
        fig['layout'][y_axis_name]['linecolor'] = "#BCCCDC"
        if len(group) > 1:
            fig['layout'][y_axis_name]['title'] = ''
        else:
            fig['layout'][y_axis_name]['title'] = group[0]

    n_results = len(search.results)
    colors = get_colors(n_colors=n_results)
    colors = {key: color for key, color in zip(search.results, colors)}
    add_label = defaultdict(lambda: True)

    # Plot the groups
    for idx, group in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        for y_key in group:
            fig.add_trace(
                Scatter(x=search.data[search.params[0]],
                        y=search.data[y_key],
                        name=y_key,
                        legendgroup=y_key,
                        showlegend=add_label[y_key],
                        mode="markers" if search.ignored_params else "lines+markers",
                        line={'color': colors[y_key]},
                        marker={'symbol': 'circle'}),
                row=row + 1,
                col=col + 1)
            add_label[y_key] = False

    # If inside a jupyter notebook then force the height based on number of rows
    if in_notebook():
        fig.update_layout(height=280 * n_rows)

    return FigureFE.from_figure(fig)


def visualize_cartesian(search: Union[Search, str],
                        title: Optional[str] = None,
                        ignore_keys: Union[None, str, Sequence[str]] = None,
                        groups: Optional[List[List[str]]] = None,
                        save_path: Optional[str] = None,
                        verbose: bool = True) -> None:
    """Display or save a parallel coordinate plot based on search results.

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.
        groups: Which result keys should be plotted on the same y-axis, rather than on separate subplots.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        verbose: Whether to print out the save location.
    """
    fig = plot_cartesian(search=search, title=title, ignore_keys=ignore_keys, groups=groups)
    fig.show(save_path=save_path, verbose=verbose, scale=5)
