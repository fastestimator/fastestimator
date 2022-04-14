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
import math
import os
import re
from collections import defaultdict
from itertools import cycle
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from natsort import humansorted
from plotly.graph_objects import Figure
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.ndimage.filters import gaussian_filter1d

from fastestimator.summary.summary import Summary, ValWithError
from fastestimator.util.util import in_notebook, prettify_metric_name, to_list, to_set


class _MetricGroup:
    """A class for wrapping the values recorded for a given metric based on its experiment id and mode.

    This class is intentionally not @traceable.
    """
    state: Dict[int, Dict[str, Dict[str, np.ndarray]]]

    def __init__(self):
        self.state = defaultdict(lambda: defaultdict(dict))  # exp_id: {mode: {ds_id: value}}

    def __getitem__(self, exp_id: int) -> Dict[str, Dict[str, np.ndarray]]:
        return self.state[exp_id]

    def add(self, exp_id: int, mode: str, ds_id: str, values: Dict[int, Any]) -> bool:
        """Add a new set of values to the metric group.

        Args:
            exp_id: The experiment id associated with these `values`.
            mode: The mode associated with these `values`.
            ds_id: The ds_id associated with these values (or empty string for None).
            values: A dictionary of time: value pairs.

        Returns:
            Whether the add was successful.
        """
        if values:
            values = list(sorted(values.items()))
            if len(values) == 1:
                # We will allow any data types if there's only one value since it will be displayed differently
                self.state[exp_id][mode][ds_id] = np.array(
                    values, dtype=None if isinstance(values[0][1], (int, float)) else object)
                return True
            else:
                # We will be plotting something over time
                val_is_object = False
                for idx, (step, elem) in enumerate(values):
                    if isinstance(elem, np.ndarray):
                        if elem.ndim == 0 or (elem.ndim == 1 and elem.shape[0] == 1):
                            elem = elem.item()
                        else:
                            # TODO - handle larger arrays somehow (maybe a heat map?)
                            return False
                    if isinstance(elem, str):
                        # Can't plot strings over time...
                        elem = [float(s) for s in re.findall(r'(\d+\.\d+|\.?\d+)', elem)]
                        if len(elem) == 1:
                            # We got an unambiguous number
                            elem = elem[0]
                        else:
                            # Can't disambiguate what should be plotted
                            return False
                    if not isinstance(elem, (int, float, ValWithError)):
                        # Can only plot numeric values over time
                        return False
                    values[idx] = (step, elem)
                    if isinstance(elem, ValWithError):
                        val_is_object = True
                if val_is_object:
                    # If some points are ValWithError, then they all need to be
                    for idx, (step, elem) in enumerate(values):
                        if isinstance(elem, (int, float)):
                            values[idx] = (step, ValWithError(elem, elem, elem))
                self.state[exp_id][mode][ds_id] = np.array(values, dtype=object if val_is_object else None)

    def ndim(self) -> int:
        """Compute how many dimensions this data require to plot.

        Returns:
            The number of dimensions this data requires to plot.
        """
        ndims = [0]
        for mode_ds_val in self.state.values():
            for _, ds_val in mode_ds_val.items():
                for _, values in ds_val.items():
                    if values.ndim in (0, 1):
                        # A singular value (this should never happen based on implementation of summary)
                        ndims.append(1)
                    elif values.ndim == 2:
                        if values.shape[0] == 1:
                            # Metrics with only 1 time point can be displayed as singular values
                            if isinstance(values[0][1], ValWithError):
                                # ValWithError, however, will always be displayed grapically
                                ndims.append(2)
                            else:
                                ndims.append(1)
                        else:
                            # A regular time vs metric value plot
                            ndims.append(2)
                    else:
                        # Time vs array value. Not supported yet.
                        ndims.append(3)
        return max(ndims)

    def get_val(self, exp_idx: int, mode: str, ds_id: str) -> Union[None, str, np.ndarray]:
        """Get the value for a given experiment id and mode.

        Args:
            exp_idx: The id of the experiment in question.
            mode: The mode under consideration.
            ds_id: The dataset id associated with this value.

        Returns:
            The value associated with the `exp_id` and `mode`, or None if no such value exists. If only a single item
            exists and it is numeric then it will be returned as a string truncated to 5 decimal places.
        """
        vals = self.state[exp_idx].get(mode, {}).get(ds_id, None)
        if vals is None:
            return vals
        if vals.ndim in (0, 1):
            item = vals.item()
            if isinstance(item, float):
                return "{:.5f}".format(item)
            return str(item)
        if vals.ndim == 2 and vals.shape[0] == 1:
            # This value isn't really time dependent
            item = vals[0][1]
            if isinstance(item, float):
                return "{:.5f}".format(item)
            return str(item)
        else:
            return vals

    def modes(self, exp_id: int) -> List[str]:
        """Get the modes supported by this group for a given experiment.

        Args:
            exp_id: The id of the experiment in question.

        Returns:
            Which modes have data for the given `exp_id`.
        """
        return list(self.state[exp_id].keys())

    def ds_ids(self, exp_id: int, mode: Optional[str] = None) -> List[str]:
        """Get the dataset ids supported by this group for a given experiment.

        If mode is provided, then only the ds_ids present for the particular mode will be returned.

        Args:
            exp_id: The id of the experiment in question.
            mode: The mode for which to consider ids, or None to consider over all modes.

        Returns:
            Which dataset ids have data for the given `exp_id` and `mode`.
        """
        if mode is None:
            mode = self.modes(exp_id)
        mode = to_list(mode)
        return [ds for md, dsv in self.state[exp_id].items() if md in mode for ds in dsv.keys()]


def plot_logs(experiments: List[Summary],
              smooth_factor: float = 0,
              ignore_metrics: Optional[Set[str]] = None,
              pretty_names: bool = False,
              include_metrics: Optional[Set[str]] = None) -> Figure:
    """A function which will plot experiment histories for comparison viewing / analysis.

    Args:
        experiments: Experiment(s) to plot.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        ignore_metrics: Any keys to ignore during plotting.
        include_metrics: A whitelist of keys to include during plotting. If None then all will be included.

    Returns:
        The handle of the pyplot figure.
    """
    # Sort to keep same colors between multiple runs of visualization
    experiments = humansorted(to_list(experiments), lambda exp: exp.name)
    n_experiments = len(experiments)
    if n_experiments == 0:
        return make_subplots()

    ignore_keys = ignore_metrics or set()
    ignore_keys = to_set(ignore_keys)
    ignore_keys |= {'epoch'}
    include_keys = to_set(include_metrics)
    # TODO: epoch should be indicated on the axis (top x axis?). Problem - different epochs per experiment.
    # TODO: figure out how ignore_metrics should interact with mode
    # TODO: when ds_id switches during training, prevent old id from connecting with new one (break every epoch?)
    ds_ids = set()
    metric_histories = defaultdict(_MetricGroup)  # metric: MetricGroup
    for idx, experiment in enumerate(experiments):
        history = experiment.history
        # Since python dicts remember insertion order, sort the history so that train mode is always plotted on bottom
        for mode, metrics in sorted(history.items(),
                                    key=lambda x: 0 if x[0] == 'train' else 1 if x[0] == 'eval' else 2 if x[0] == 'test'
                                    else 3 if x[0] == 'infer' else 4):
            for metric, step_val in metrics.items():
                base_metric, ds_id, *_ = f'{metric}|'.split('|')  # Plot acc|ds1 and acc|ds2 on same acc graph
                if len(step_val) == 0:
                    continue  # Ignore empty metrics
                if metric in ignore_keys or base_metric in ignore_keys:
                    continue
                # Here we intentionally check against metric and not base_metric. If user wants to display per-ds they
                #  can specify that in their include list: --include mcc 'mcc|usps'
                if include_keys and metric not in include_keys:
                    continue
                metric_histories[base_metric].add(idx, mode, ds_id, step_val)
                ds_ids.add(ds_id)

    metric_list = list(sorted(metric_histories.keys()))
    if len(metric_list) == 0:
        return make_subplots()
    ds_ids = humansorted(ds_ids)  # Sort them to have consistent ordering (and thus symbols) between plot runs
    n_plots = len(metric_list)

    # Non-Shared legends aren't supported yet. If they get supported then maybe can have that feature here too.
    #  https://github.com/plotly/plotly.js/issues/5099
    #  https://github.com/plotly/plotly.js/issues/5098

    # map the metrics into an n x n grid, then remove any extra columns. Final grid will be n x m with m <= n
    n_rows = math.ceil(math.sqrt(n_plots))
    n_cols = math.ceil(n_plots / n_rows)
    metric_grid_location = {}
    nd1_metrics = []
    idx = 0
    for metric in metric_list:
        if metric_histories[metric].ndim() == 1:
            # Delay placement of the 1D plots until the end
            nd1_metrics.append(metric)
        else:
            metric_grid_location[metric] = (idx // n_cols, idx % n_cols)
            idx += 1
    for metric in nd1_metrics:
        metric_grid_location[metric] = (idx // n_cols, idx % n_cols)
        idx += 1
    titles = [k for k, v in sorted(list(metric_grid_location.items()), key=lambda e: e[1][0] * n_cols + e[1][1])]
    if pretty_names:
        titles = [prettify_metric_name(title) for title in titles]

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, shared_xaxes='all')
    fig.update_layout({'plot_bgcolor': '#FFF',
                       'hovermode': 'closest',
                       'margin': {'t': 50},
                       'modebar': {'add': ['hoverclosest', 'hovercompare'],
                                   'remove': ['select2d', 'lasso2d']},
                       'legend': {'tracegroupgap': 5,
                                  'font': {'size': 11}}})

    # Set x-labels
    for idx, metric in enumerate(titles, start=1):
        plotly_idx = idx if idx > 1 else ""
        x_axis_name = f'xaxis{plotly_idx}'
        y_axis_name = f'yaxis{plotly_idx}'
        if metric_histories[metric].ndim() > 1:
            fig['layout'][x_axis_name]['title'] = 'Steps'
            fig['layout'][x_axis_name]['showticklabels'] = True
            fig['layout'][x_axis_name]['linecolor'] = "#BCCCDC"
            fig['layout'][y_axis_name]['linecolor'] = "#BCCCDC"
        else:
            # Put blank data onto the axis to instantiate the domain
            row, col = metric_grid_location[metric][0], metric_grid_location[metric][1]
            fig.add_annotation(text='', showarrow=False, row=row + 1, col=col + 1)
            # Hide the axis stuff
            fig['layout'][x_axis_name]['showgrid'] = False
            fig['layout'][x_axis_name]['zeroline'] = False
            fig['layout'][x_axis_name]['visible'] = False
            fig['layout'][y_axis_name]['showgrid'] = False
            fig['layout'][y_axis_name]['zeroline'] = False
            fig['layout'][y_axis_name]['visible'] = False

    colors = sns.hls_palette(n_colors=n_experiments, s=0.95) if n_experiments > 10 else sns.color_palette("colorblind")
    alpha_colors = [f'rgba({int(rgb[0]*256)},{int(rgb[1]*256)},{int(rgb[2]*256)},0.3)' for rgb in colors]
    colors = [f'rgb({int(rgb[0]*256)},{int(rgb[1]*256)},{int(rgb[2]*256)})' for rgb in colors]
    color_offset = defaultdict(lambda: 0)
    # If there is only 1 experiment, we will use alternate colors based on mode
    if n_experiments == 1:
        color_offset['eval'] = 1
        color_offset['test'] = 2
        color_offset['infer'] = 3

    # exp_id : {mode: {ds_id: {type: True}}}
    add_label = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: True))))
    # {row: {col: (x, y)}}
    ax_text = defaultdict(lambda: defaultdict(lambda: (0.0, 0.9)))  # Where to put the text on a given axis
    # Set up ds_id markers. The empty ds_id will have no extra marker. After that there are 4 configurations of 3-arm
    # marker, followed by 'x', '+', '*', and pound. After that it will just repeat the symbol set.
    ds_id_markers = [None, 37, 38, 39, 40, 34, 33, 35, 36]  # https://plotly.com/python/marker-style/
    ds_id_unicode = [None, "\U00002144", "\U0001d5b8", "\U00002919", "\U0000291a", "\U000000d7", "\U0000002b",
                     "\U00002733", "\U00002317"]  # Symbols located using https://shapecatcher.com/index.html
    ds_id_markers = {k: v for k, v in zip(ds_ids, cycle(ds_id_markers))}
    ds_id_unicode = {k: '' if v is None else f'{v} ' for k, v in zip(ds_ids, cycle(ds_id_unicode))}
    # Plotly doesn't support z-order, so delay insertion until all the plots are figured out:
    # https://github.com/plotly/plotly.py/issues/2345
    z_order = defaultdict(list)  # {order: [(plotly element, row, col), ...]}

    # Figure out the legend ordering
    legend_order = []
    for exp_idx, experiment in enumerate(experiments):
        for metric, group in metric_histories.items():
            for mode in group.modes(exp_idx):
                for ds_id in group.ds_ids(exp_idx, mode):
                    ds_title = f"{ds_id} " if ds_id else ''
                    title = f"{experiment.name} ({ds_title}{mode})" if n_experiments > 1 else f"{ds_title}{mode}"
                    legend_order.append(title)
    legend_order.sort()
    legend_order = {legend: order for order, legend in enumerate(legend_order)}

    # Actually do the plotting
    for exp_idx, experiment in enumerate(experiments):
        for metric, group in metric_histories.items():
            row, col = metric_grid_location[metric][0], metric_grid_location[metric][1]
            if group.ndim() == 1:
                # Single value
                for mode in group.modes(exp_idx):
                    for ds_id in group.ds_ids(exp_idx, mode):
                        ds_title = f"{ds_id} " if ds_id else ''
                        prefix = f"{experiment.name} ({ds_title}{mode})" if n_experiments > 1 else f"{ds_title}{mode}"
                        plotly_idx = row*n_cols+col+1 if row*n_cols+col+1 > 1 else ''
                        fig.add_annotation(text=f"{prefix}: {group.get_val(exp_idx, mode, ds_id)}",
                                           font={'color': colors[exp_idx + color_offset[mode]]},
                                           showarrow=False,
                                           xref=f'x{plotly_idx} domain',
                                           xanchor='left',
                                           x=ax_text[row][col][0],
                                           yref=f'y{plotly_idx} domain',
                                           yanchor='top',
                                           y=ax_text[row][col][1],
                                           exclude_empty_subplots=False)
                        ax_text[row][col] = (ax_text[row][col][0], ax_text[row][col][1] - 0.1)
                        if ax_text[row][col][1] < 0:
                            ax_text[row][col] = (ax_text[row][col][0] + 0.5, 0.9)
            elif group.ndim() == 2:
                for mode, dsv in group[exp_idx].items():
                    color = colors[exp_idx + color_offset[mode]]
                    for ds_id, data in dsv.items():
                        ds_title = f"{ds_id} " if ds_id else ''
                        title = f"{experiment.name} ({ds_title}{mode})" if n_experiments > 1 else f"{ds_title}{mode}"
                        if data.shape[0] < 2:
                            x = data[0][0]
                            y = data[0][1]
                            y_min = None
                            y_max = None
                            if isinstance(y, ValWithError):
                                y_min = y.y_min
                                y_max = y.y_max
                                y = y.y
                            marker_style = 'circle' if mode == 'train' else 'diamond' if mode == 'eval' \
                                else 'square' if mode == 'test' else 'hexagram'
                            limit_data = [(y_max, y_min)] if y_max is not None and y_min is not None else None
                            tip_text = "%{x}: (%{customdata[1]:.3f}, %{y:.3f}, %{customdata[0]:.3f})" if \
                                limit_data is not None else "%{x}: %{y:.3f}"
                            error_y = None if limit_data is None else {'type': 'data',
                                                                       'symmetric': False,
                                                                       'array': [y_max-y],
                                                                       'arrayminus': [y-y_min]}
                            z_order[2].append((go.Scatter(x=[x],
                                                          y=[y],
                                                          name=f"{ds_id_unicode[ds_id]}{title}",
                                                          legendgroup=title,
                                                          customdata=limit_data,
                                                          hovertemplate=tip_text,
                                                          mode='markers',
                                                          marker={'color': color,
                                                                  'size': 12,
                                                                  'symbol': marker_style},
                                                          error_y=error_y,
                                                          showlegend=add_label[exp_idx][mode][ds_id]['patch'],
                                                          legendrank=legend_order[title]),
                                              row,
                                              col))
                            add_label[exp_idx][mode][ds_id]['patch'] = False
                            if ds_id and ds_id_markers[ds_id]:
                                # TODO - get this overlayed in legend. maybe need to custom draw the shape?
                                #  https://plotly.com/python/shapes/
                                z_order[3].append((go.Scatter(x=[x],
                                                              y=[y],
                                                              name=title,
                                                              legendgroup=title,
                                                              hoverinfo='skip',
                                                              mode='markers',
                                                              marker={'color': color,
                                                                      'size': 8,
                                                                      'line': {'width': 1.5,
                                                                               'color': 'White'},
                                                                      'symbol': ds_id_markers[ds_id]},
                                                              showlegend=False),
                                                   row, col))
                        else:
                            # We can draw a line
                            y = data[:, 1]
                            y_min = None
                            y_max = None
                            if isinstance(y[0], ValWithError):
                                y = np.stack(y)
                                y_min = y[:, 0]
                                y_max = y[:, 2]
                                y = y[:, 1]
                                if smooth_factor != 0:
                                    y_min = gaussian_filter1d(y_min, sigma=smooth_factor)
                                    y_max = gaussian_filter1d(y_max, sigma=smooth_factor)
                            # TODO - for smoothed lines, plot original data in background but greyed out
                            if smooth_factor != 0:
                                y = gaussian_filter1d(y, sigma=smooth_factor)
                            x = data[:, 0]
                            linestyle = 'solid' if mode == 'train' else 'dash' if mode == 'eval' else 'dot' if \
                                mode == 'test' else 'dashdot'
                            limit_data = [(mx, mn) for mx, mn in zip(y_max, y_min)] if y_max is not None and y_min is \
                                not None else None
                            tip_text = "%{x}: (%{customdata[1]:.3f}, %{y:.3f}, %{customdata[0]:.3f})" if \
                                limit_data is not None else "%{x}: %{y:.3f}"
                            z_order[1].append((go.Scatter(x=x,
                                                          y=y,
                                                          name=title,
                                                          legendgroup=title,
                                                          mode="lines+markers" if ds_id_markers[ds_id] else 'lines',
                                                          marker={'color': color,
                                                                  'size': 8,
                                                                  'line': {'width': 2,
                                                                           'color': 'DarkSlateGrey'},
                                                                  'maxdisplayed': 10,
                                                                  'symbol': ds_id_markers[ds_id]},
                                                          line={'dash': linestyle,
                                                                'color': color},
                                                          customdata=limit_data,
                                                          hovertemplate=tip_text,
                                                          showlegend=add_label[exp_idx][mode][ds_id]['line'],
                                                          legendrank=legend_order[title]),
                                              row,
                                              col))
                            add_label[exp_idx][mode][ds_id]['line'] = False
                            if limit_data is not None:
                                z_order[0].append((go.Scatter(x=x,
                                                              y=y_max,
                                                              mode='lines',
                                                              line={'width': 0},
                                                              legendgroup=title,
                                                              showlegend=False,
                                                              hoverinfo='skip'),
                                                   row,
                                                   col))
                                z_order[0].append((go.Scatter(x=x,
                                                              y=y_min,
                                                              mode='lines',
                                                              line={'width': 0},
                                                              fillcolor=alpha_colors[exp_idx+color_offset[mode]],
                                                              fill='tonexty',
                                                              legendgroup=title,
                                                              showlegend=False,
                                                              hoverinfo='skip'),
                                                  row,
                                                  col))
            else:
                # Some kind of image or matrix. Not implemented yet.
                pass
    for z in sorted(list(z_order.keys())):
        plts = z_order[z]
        for plt, row, col in plts:
            fig.add_trace(plt, row=row+1, col=col+1)

    # If inside a jupyter notebook then force the height based on number of rows
    if in_notebook():
        fig.update_layout(height=280 * n_rows)
    return fig


def visualize_logs(experiments: List[Summary],
                   save_path: str = None,
                   smooth_factor: float = 0,
                   pretty_names: bool = False,
                   ignore_metrics: Optional[Set[str]] = None,
                   include_metrics: Optional[Set[str]] = None,
                   verbose: bool = True):
    """A function which will save or display experiment histories for comparison viewing / analysis.

    Args:
        experiments: Experiment(s) to plot.
        save_path: The path where the figure should be saved, or None to display the figure to the screen.
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
        pretty_names: Whether to modify the metric names in graph titles (True) or leave them alone (False).
        ignore_metrics: Any metrics to ignore during plotting.
        include_metrics: A whitelist of metric keys (None whitelists all keys).
        verbose: Whether to print out the save location.
    """
    fig = plot_logs(experiments,
                    smooth_factor=smooth_factor,
                    pretty_names=pretty_names,
                    ignore_metrics=ignore_metrics,
                    include_metrics=include_metrics)
    config = {
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            'height': None,
            'width': None,
            'filename': 'parse_logs',
            'scale': 5  # Multiply title/legend/axis/canvas sizes by this factor (high resolution save)
        }}
    if save_path is None:
        fig.show(config=config)
    else:
        save_path = os.path.normpath(save_path)
        root_dir = os.path.dirname(save_path)
        if root_dir == "":
            root_dir = "."
        os.makedirs(root_dir, exist_ok=True)
        save_file = os.path.join(root_dir, os.path.basename(save_path) or 'parse_logs.html')
        config['toImageButtonOptions']['filename'] = os.path.splitext(os.path.basename(save_file))[0]
        ext = os.path.splitext(save_file)[1]
        if ext == '':
            ext = '.html'
            save_file = save_file + ext  # Use html by default
        if verbose:
            print("Saving to {}".format(save_file))
        if ext == '.html':
            fig.write_html(save_file, config=config)
        else:
            fig.write_image(save_file, width=1920, height=1080, scale=5)
