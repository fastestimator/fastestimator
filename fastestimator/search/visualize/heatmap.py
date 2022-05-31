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
from typing import Optional, Sequence, Union

from natsort import humansorted
from plotly.graph_objects import Heatmap
from plotly.subplots import make_subplots

from fastestimator.search.search import Search
from fastestimator.search.visualize.vis_util import SearchData, _load_search_file
from fastestimator.util.base_util import in_notebook, FigureFE


def _heatmap_supports_data(data: SearchData, throw_on_invalid: bool = True) -> bool:
    """Determine whether heatmap visualization supports the given data.

    Args:
        data: The data to be visualized.
        throw_on_invalid: Whether to throw an exception if the data is invalid.

    Returns:
        True iff the data can be visualized via heatmaps.
    """
    if len(data.params) != 2:
        if not throw_on_invalid:
            return False
        raise ValueError(f"Heatmap only supports exactly 2 params, but found {len(data.params)}: {data.params}")
    if len(data.results) == 0:
        if not throw_on_invalid:
            return False
        raise ValueError("Heatmap requires 1+ results, but found none.")
    for result in data.results:
        if result in data.categorical_maps:
            if not throw_on_invalid:
                return False
            raise ValueError(f"Heatmap only supports numeric results, but found categorical results for key: {result}")
    if data.ignored_params:
        if not throw_on_invalid:
            return False
        raise ValueError("Heatmaps do not support ignoring parameters which have more than 1 value")
    return True


def plot_heatmap(search: Union[Search, str],
                 title: Optional[str] = None,
                 ignore_keys: Union[None, str, Sequence[str]] = None) -> FigureFE:
    """Draw a colormap plot based on search results.

    Requires exactly 2 params and 1 result (after accounting for the ignore_keys).

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.

    Returns:
        A plotly figure instance.
    """
    if isinstance(search, str):
        search = _load_search_file(search)
    if title is None:
        title = search.name
    reverse_colors = search.best_mode == 'min'
    search = SearchData(search=search, ignore_keys=ignore_keys)
    _heatmap_supports_data(search)

    # Convert all params to be categorical
    x = [search.to_category(key=search.params[0], val=e) for e in search.data[search.params[0]]]
    x_labels = humansorted(set(x))

    y = [search.to_category(key=search.params[1], val=e) for e in search.data[search.params[1]]]
    y_labels = humansorted(set(y))

    # Map the metrics into an n x n grid, then remove any extra columns. Final grid will be n x m with n <= m
    n_plots = len(search.results)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    vertical_gap = 0.15 / n_rows
    horizontal_gap = 0.2 / n_cols

    # Get basic plot layout
    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        subplot_titles=search.results,
                        shared_xaxes='all',
                        shared_yaxes='all',
                        vertical_spacing=vertical_gap,
                        horizontal_spacing=horizontal_gap,
                        x_title=search.params[0],
                        y_title=search.params[1])
    fig.update_layout({'title': title,
                       'title_x': 0.5,
                       })

    # Fill in the penultimate row x-labels when the last row has empty columns
    for idx in range((n_plots % n_cols) or n_cols, n_cols):
        plotly_idx = max((n_rows - 2) * n_cols, 0) + idx + 1
        x_axis_name = f'xaxis{plotly_idx}'
        fig['layout'][x_axis_name]['showticklabels'] = True

    # Ensure the categories are in the right order
    fig['layout']['xaxis']['categoryarray'] = x_labels
    fig['layout']['yaxis']['categoryarray'] = y_labels

    plot_height = (1 - (n_rows - 1) * vertical_gap) / n_rows
    plot_width = (1 - (n_cols - 1) * horizontal_gap) / n_cols

    # Plot the groups
    for idx, plot in enumerate(search.results):
        row = idx // n_cols
        col = idx % n_cols
        fig.add_trace(Heatmap(x=x,
                              y=y,
                              z=search.data[plot],
                              colorscale="Viridis",
                              reversescale=reverse_colors,
                              colorbar={'len': plot_height,
                                        'lenmode': 'fraction',
                                        'yanchor': 'top',
                                        'y': 1 - row * (plot_height + vertical_gap),
                                        'xanchor': 'left',
                                        'x': col * (plot_width + horizontal_gap) + plot_width},
                              name="",
                              hovertemplate=search.params[0] + ": %{x}<br>" + search.params[1] + ": %{y}<br>" +
                                            plot + ": %{z}",
                              hoverongaps=False),
                      row=row + 1,
                      col=col + 1)

        # Make sure that the image aspect ratio doesn't get messed up
        x_axis_name = fig.get_subplot(row=row + 1, col=col + 1).xaxis.plotly_name
        y_axis_name = fig.get_subplot(row=row + 1, col=col + 1).yaxis.plotly_name
        fig['layout'][x_axis_name]['scaleanchor'] = 'x'
        fig['layout'][x_axis_name]['scaleratio'] = 1
        fig['layout'][x_axis_name]['constrain'] = 'domain'
        fig['layout'][y_axis_name]['scaleanchor'] = 'x'
        fig['layout'][y_axis_name]['constrain'] = 'domain'

    # If inside a jupyter notebook then force the height based on number of rows
    if in_notebook():
        fig.update_layout(height=500 * max(1.0, len(y_labels)/5.0) * n_rows)
        fig.update_layout(width=500 * max(1.0, len(x_labels)/5.0) * n_cols)

    return FigureFE.from_figure(fig)


def visualize_heatmap(search: Union[Search, str],
                      title: Optional[str] = None,
                      ignore_keys: Union[None, str, Sequence[str]] = None,
                      save_path: Optional[str] = None,
                      verbose: bool = True) -> None:
    """Display or save a parallel coordinate plot based on search results.

    Args:
        search: The search results (in memory or path to disk file) to be visualized.
        title: The plot title to use.
        ignore_keys: Which keys in the params/results should be ignored.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        verbose: Whether to print out the save location.
    """
    fig = plot_heatmap(search=search, title=title, ignore_keys=ignore_keys)
    fig.show(save_path=save_path, verbose=verbose, scale=3)
