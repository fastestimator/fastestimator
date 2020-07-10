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
from matplotlib.gridspec import GridSpec

from fastestimator.util.util import show_image, to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class ImgData(OrderedDict):
    """A container for image related data.

    This class is intentionally not @traceable.

    This class is useful for automatically laying out collections of images for comparison and visualization.

    ```python
    d = fe.util.ImgData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
    fig = d.paint_figure()
    plt.show()

    img = 0.5*np.ones((4, 32, 32, 3))
    mask = np.zeros_like(img)
    mask[0, 10:20, 10:30, :] = [1, 0, 0]
    mask[1, 5:15, 5:20, :] = [0, 1, 0]
    bbox = np.array([[[3,7,10,6,'box1'], [20,20,8,8,'box2']]]*4)
    d = fe.util.ImgData(y=tf.ones((4,)), x=[img, mask, bbox])
    fig = d.paint_figure()
    plt.show()
    ```

    Args:
        colormap: What colormap to use when rendering greyscale images. A good colorization option is 'inferno'.
        **kwargs: image_title / image pairs for visualization. Images with the same batch dimensions will be laid out
            side-by-side, with earlier kwargs entries displayed further to the left. The value part of the key/value
            pair can be a list of tensors, in which case the elements of the list are overlaid. This can be useful for
            displaying masks and bounding boxes on top of images. In such cases, the largest image should be put as the
            first entry in the list. Bounding boxes should be shaped like (batch, n_boxes, box), where each box is
            formatted like (x0, y0, width, height[, label]).

    Raises:
        AssertionError: If a list of Tensors is provided as an input, but that list has an inconsistent batch dimension.
    """
    n_elements: Dict[int, List[str]]

    def __init__(self, colormap: str = "Greys", **kwargs: Union[Tensor, List[Tensor]]) -> None:
        self.n_elements = {}  # Not a default dict b/c that complicates the computations later
        self.colormap = colormap
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Union[Tensor, List[Tensor]]):
        # Convert all values into a list for consistency
        value = to_list(value)
        batch_size = value[0].shape[0]
        for elem in value[1:]:
            assert elem.shape[0] == batch_size, "Provided item has an inconsistent batch size"
        super().__setitem__(key, value)
        self.n_elements.setdefault(batch_size, []).append(key)

    def __delitem__(self, key: str):
        super().__delitem__(key)
        for k, lst in self.n_elements.items():
            lst.remove(key)
            if len(lst) == 0:
                del self.n_elements[k]

    def _to_grid(self) -> List[List[Tuple[str, np.ndarray]]]:
        """Convert the elements of ImgData into a grid view.

        One row in the grid is generated for each unique batch dimension present within the ImgData. Each column in the
        grid is a tensor with batch dimension matching the current row. Columns are given in the order they were input
        into the ImgData constructor.

        Returns:
            The ImgData arranged as a grid, with entries in the grid as (key, value) pairs.
        """
        sorted_sections = sorted(self.n_elements.keys())
        return [[(key, self[key]) for key in self.n_elements[n_rows]] for n_rows in sorted_sections]

    def _n_rows(self) -> int:
        """Computes how many rows are present in the ImgData grid.

        Returns:
            The number of rows in the ImgData grid.
        """
        return len(self.n_elements)

    def _n_cols(self) -> int:
        """Computes how many columns are present in the ImgData grid.

        Returns:
            The number of columns in the ImgData grid.
        """
        return max((len(elem) for elem in self.n_elements.values()))

    @staticmethod
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

    @staticmethod
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
        # For overlay values consider the zeroth element for the shape
        row = [self[key][0] for key in self.n_elements[keys[row]]]
        widths = [(0, ImgData._shape_to_width(row[0].shape, min_width=min_width))]
        for img in row[1:]:
            widths.append(
                (widths[-1][1] + gap, widths[-1][1] + gap + ImgData._shape_to_width(img.shape, min_width=min_width)))
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
        # For overlay values consider the zeroth element for the shape
        rows = [[self[key][0] for key in self.n_elements[keys[row]]] for row in range(self._n_rows())]
        heights = [
            max((ImgData._shape_to_height(elem.shape, min_height=min_height) for elem in rows[i]))
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
        """Visualize the current ImgData entries in a matplotlib figure.

        ```python
        d = fe.util.ImgData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
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
                    img_stack = [elem[batch_idx] for elem in row[col_idx][1]]
                    for idx, img in enumerate(img_stack):
                        show_image(img,
                                   axis=ax,
                                   fig=fig,
                                   title=row[col_idx][0] if (batch_idx == 0 and idx == 0) else None,
                                   stack_depth=idx,
                                   color_map=self.colormap)
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def paint_numpy(self,
                    height_gap: int = 100,
                    min_height: int = 200,
                    width_gap: int = 50,
                    min_width: int = 200,
                    dpi: int = 96) -> np.ndarray:
        """Visualize the current ImgData entries into an image stored in a numpy array.

        ```python
        d = fe.util.ImgData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))
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
            A numpy array with dimensions (1, height, width, 3) containing an image representation of this ImgData.
        """
        fig = self.paint_figure(height_gap=height_gap,
                                min_height=min_height,
                                width_gap=width_gap,
                                min_width=min_width,
                                dpi=dpi)
        canvas = plt_backend_agg.FigureCanvasAgg(fig)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data = data.reshape([h, w, 4])[:, :, 0:3]
        plt.close(fig)
        return np.stack([data])  # Add a batch dimension
