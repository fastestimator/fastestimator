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
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from fastestimator.backend.to_number import to_number
from matplotlib.gridspec import GridSpec

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def show_image(im: Union[np.ndarray, Tensor],
               axis: plt.Axes = None,
               fig: plt.Figure = None,
               title: Optional[str] = None,
               color_map: str = "inferno") -> Optional[plt.Figure]:
    """Plots a given image onto an axis.

    Args:
        axis: The matplotlib axis to plot on, or None for a new plot.
        fig: A reference to the figure to plot on, or None if new plot.
        im: The image to display (width X height).
        title: A title for the image.
        color_map: Which colormap to use for greyscale images.
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    axis.axis('off')
    # Compute width of axis for text font size
    bbox = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    space = min(width, height)
    if not hasattr(im, 'shape') or len(im.shape) < 2:
        # text data
        im = to_number(im)
        if hasattr(im, 'shape') and len(im.shape) == 1:
            im = im[0]
        im = im.item()
        if isinstance(im, bytes):
            im = im.decode('utf8')
        text = "{}".format(im)
        axis.text(0.5,
                  0.5,
                  im,
                  ha='center',
                  transform=axis.transAxes,
                  va='center',
                  wrap=False,
                  family='monospace',
                  fontsize=min(45, space // len(text)))
    else:
        if isinstance(im, torch.Tensor) and len(im.shape) > 2:
            # Move channel first to channel last
            channels = list(range(len(im.shape)))
            channels.append(channels.pop(0))
            im = im.permute(*channels)
        # image data
        im = to_number(im)
        if np.issubdtype(im.dtype, np.integer):
            # im is already in int format
            im = im.astype(np.uint8)
        elif np.max(im) <= 1 and np.min(im) >= 0:  # im is [0,1]
            im = (im * 255).astype(np.uint8)
        elif np.min(im) >= -1 and np.max(im) <= 1:  # im is [-1, 1]
            im = ((im + 1) * 127.5).astype(np.uint8)
        else:  # im is in some arbitrary range, probably due to the Normalize Op
            ma = abs(np.max(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            mi = abs(np.min(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            im = (((im + mi) / (ma + mi)) * 255).astype(np.uint8)
        # matplotlib doesn't support (x,y,1) images, so convert them to (x,y)
        if len(im.shape) == 3 and im.shape[2] == 1:
            im = np.reshape(im, (im.shape[0], im.shape[1]))
        if len(im.shape) == 2:
            axis.imshow(im, cmap=plt.get_cmap(name=color_map))
        else:
            axis.imshow(im)
    if title is not None:
        axis.set_title(title, fontsize=min(20, 1 + width // len(title)), family='monospace')
    return fig


def _shape_to_width(shape: Tuple[int], min_width=200) -> int:
    """Decide the width of an image for visualization.

    Args:
        shape: The shape of the image.
        min_width: The minimum desired width for visualization.

    Returns:
        The maximum between the width specified by `shape` and the given `min_width` value.
    """
    if len(shape) < 2:
        # text field, use default width
        pass
    elif len(shape) == 2:
        # image field: width x height
        min_width = max(shape[0], min_width)
    else:
        # image field: batch x width x height
        min_width = max(shape[1], min_width)
    return min_width


def _shape_to_height(shape: Tuple[int], min_height=200) -> int:
    """Decide the height of an image for visualization.

    Args:
        shape: The shape of the image.
        min_height: The minimum desired width for visualization.

    Returns:
        The maximum between the height specified by `shape` and the given `min_height` value.
    """
    if len(shape) < 2:
        # text field, use default width
        pass
    elif len(shape) == 2:
        # image field: width x height
        min_height = max(shape[0], min_height)
    else:
        # image field: batch x width x height
        min_height = max(shape[1], min_height) * shape[0]
    return min_height


class XaiData(OrderedDict):
    """A container for xai related data.

    This class is useful for automatically laying out collections of images for comparison and visualization.

    ```python
    d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
    fig = d.paint_figure()
    plt.show()
    ```

    Args:
        **kwargs: image_title / image pairs for visualization. Images with the same batch dimensions will be laid out
            side-by-side, with earlier kwargs entries displayed further to the left.
    """
    n_elements: Dict[int, List[str]]

    def __init__(self, **kwargs: Tensor) -> None:
        self.n_elements = {}  # Not a default dict b/c that complicates the computations later
        # TODO - grouping text keys into single box (true value, predicted value, confidence, etc.)
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Tensor):
        super().__setitem__(key, value)
        self.n_elements.setdefault(value.shape[0], []).append(key)

    def __delitem__(self, key: str):
        super().__delitem__(key)
        for k, lst in self.n_elements.items():
            lst.remove(key)
            if len(lst) == 0:
                del self.n_elements[k]

    def _to_grid(self) -> List[List[Tuple[str, np.ndarray]]]:
        """Convert the elements of XaiData into a grid view.

        One row in the grid is generated for each unique batch dimension present within the XaiData. Each column in the
        grid is a tensor with batch dimension matching the current row. Columns are given in the order they were input
        into the XaiData constructor.

        Returns:
            The XaiData arranged as a grid, with entries in the grid as (key, value) pairs.
        """
        sorted_sections = sorted(self.n_elements.keys())
        return [[(key, self[key]) for key in self.n_elements[n_rows]] for n_rows in sorted_sections]

    def _n_rows(self) -> int:
        """Computes how many rows are present in the XaiData grid.

        Returns:
            The number of rows in the XaiData grid.
        """
        return len(self.n_elements)

    def _n_cols(self) -> int:
        """Computes how many columns are present in the XaiData grid.

        Returns:
            The number of columns in the XaiData grid.
        """
        return max((len(elem) for elem in self.n_elements.values()))

    def _widths(self, row: int, gap: int = 50, min_width: int = 200) -> List[Tuple[int, int]]:
        """Get the display widths of a particular row.

        Args:
            row: The row to measure.
            gap: How much space to allow between each column.
            min_width: The minimum width for a column.

        Returns:
            A list of (x1, x2) coordinates marking the beginning and end coordinates of each column in the `row`.
        """
        keys = list(sorted(self.n_elements.keys()))
        row = [self[key] for key in self.n_elements[keys[row]]]
        widths = [(0, _shape_to_width(row[0].shape, min_width=min_width))]
        for img in row[1:]:
            widths.append((widths[-1][1] + gap, widths[-1][1] + gap + _shape_to_width(img.shape, min_width=min_width)))
        return widths

    def _total_width(self, gap: int = 50, min_width: int = 200) -> int:
        """Get the total width necessary for the image by considering the widths of each row.

        Args:
            gap: The horizontal space between each column in the grid.
            min_width: The minimum width of a column.

        Returns:
            The total width of the image.
        """
        return max(
            (self._widths(row, gap=gap, min_width=min_width)[-1][-1] for row in range(len(self.n_elements.keys()))))

    def _heights(self, gap: int = 100, min_height: int = 200) -> List[Tuple[int, int]]:
        """Get the display heights of each row.

        Args:
            gap: How much space to allow between each row.
            min_height: The minimum height for a row.

        Returns:
            A list of (y1, y2) coordinates marking the top and bottom coordinates of each row in the grid.
        """
        keys = list(sorted(self.n_elements.keys()))
        rows = [[self[key] for key in self.n_elements[keys[row]]] for row in range(self._n_rows())]
        heights = [
            max((_shape_to_height(elem.shape, min_height=min_height) for elem in rows[i]))
            for i in range(self._n_rows())
        ]
        offset = 10
        result = [(offset, heights[0] + offset)]
        for height in heights[1:]:
            result.append((result[-1][1] + gap, result[-1][1] + gap + height))
        return result

    def _total_height(self, gap: int = 100, min_height: int = 200) -> int:
        """Get the total height necessary for the image by considering the heights of each row.

        Args:
            gap: The vertical space between each row in the grid.
            min_height: The minimum height of a row.

        Returns:
            The total height of the image.
        """
        heights = self._heights(gap=gap, min_height=min_height)
        # Add some space at the top for the labels
        return heights[-1][1] + 30

    def _batch_size(self, row: int) -> int:
        """Get the batch size associated with the given `row`.

        Args:
            row: The row for which to report the batch size.

        Returns:
            The batch size of all of the entries in the row.
        """
        return sorted(self.n_elements.keys())[row]

    def paint_figure(self,
                     height_gap: int = 100,
                     min_height: int = 200,
                     width_gap: int = 50,
                     min_width: int = 200,
                     dpi: int = 96,
                     save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the current XaiData entries in a matplotlib figure.

        ```python
        d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
        fig = d.paint_figure()
        plt.show()
        ```

        Args:
            height_gap: How much space to put between each row.
            min_height: The minimum height of a row.
            width_gap: How much space to put between each column.
            min_width: The minimum width of a column.
            dpi: The resolution of the image to display.
            save_path: If provided, the figure will be saved to the given path.

        Returns:
            The handle to the generated matplotlib figure.
        """
        total_width = self._total_width(gap=width_gap, min_width=min_width)
        total_height = self._total_height(gap=height_gap, min_height=min_height)

        fig = plt.figure(figsize=(total_width / dpi, total_height / dpi), dpi=dpi)

        grid = self._to_grid()
        # TODO - elements with batch size = 1 should be laid out in a grid like for plotting
        for row_idx, (start_height, end_height) in enumerate(self._heights(gap=height_gap, min_height=min_height)):
            row = grid[row_idx]
            batch_size = self._batch_size(row_idx)
            gs = GridSpec(nrows=batch_size,
                          ncols=total_width,
                          figure=fig,
                          left=0.0,
                          right=1.0,
                          bottom=start_height / total_height,
                          top=end_height / total_height,
                          hspace=0.05,
                          wspace=0.0)
            for batch_idx in range(batch_size):
                for col_idx, width in enumerate(self._widths(row=row_idx, gap=width_gap, min_width=min_width)):
                    ax = fig.add_subplot(gs[batch_idx, width[0]:width[1]])
                    show_image(row[col_idx][1][batch_idx],
                               axis=ax,
                               fig=fig,
                               title=row[col_idx][0] if batch_idx == 0 else None)
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def paint_numpy(self,
                    height_gap: int = 100,
                    min_height: int = 200,
                    width_gap: int = 50,
                    min_width: int = 200,
                    dpi: int = 96) -> np.ndarray:
        """Visualize the current XaiData entries into an image stored in a numpy array.

        ```python
        d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
        img = d.paint_numpy()
        plt.imshow(img[0])
        plt.show()
        ```

        Args:
            height_gap: How much space to put between each row.
            min_height: The minimum height of a row.
            width_gap: How much space to put between each column.
            min_width: The minimum width of a column.
            dpi: The resolution of the image to display.

        Returns:
            A numpy array with dimensions (1, height, width, 3) containing an image representation of this XaiData.
        """
        fig = self.paint_figure(height_gap=height_gap,
                                min_height=min_height,
                                width_gap=width_gap,
                                min_width=min_width,
                                dpi=dpi)
        # TODO - verify in jupyter notebook
        canvas = plt_backend_agg.FigureCanvasAgg(fig)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data = data.reshape([h, w, 4])[:, :, 0:3]
        plt.close(fig)
        return np.stack([data])  # Add a batch dimension
