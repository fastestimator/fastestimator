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
from itertools import zip_longest
from typing import Optional, Sequence, Tuple, TypeVar, TYPE_CHECKING, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
from plotly.colors import sample_colorscale
from plotly.graph_objects import Figure, Image
from plotly.subplots import make_subplots

from fastestimator.util.base_util import to_list, FigureFE, in_notebook, get_colors
from fastestimator.util.util import to_number

if TYPE_CHECKING:
    import tensorflow as tf

    Tensor = TypeVar('Tensor', np.ndarray, tf.Tensor, torch.Tensor)
    BoundingBox = TypeVar('BoundingBox',
                          Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]],
                          Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], str])
    KeyPoint = TypeVar('KeyPoint',
                       Tuple[Union[int, float], Union[int, float]],
                       Tuple[Union[int, float], Union[int, float], str])


class Display(ABC):
    @abstractmethod
    def prepare(self, **kwargs) -> FigureFE:
        raise NotImplementedError()

    def show(self,
             save_path: Optional[str] = None,
             verbose: bool = True,
             scale: int = 1,
             interactive: bool = False) -> None:
        """A function which will save or display the image as a plotly figure.

        Args:
            save_path: The path where the figure should be saved, or None to display the figure to the screen.
            verbose: Whether to print out the save location.
            scale: A scaling factor to apply when exporting to static images (to increase resolution).
            interactive: Whether the figure should be interactive or static. This is only applicable when
                save_path is None and when running inside a jupyter notebook. The advantage is that the file size of the
                resulting jupyter notebook can be dramatically reduced.
        """
        fig = self.prepare()
        fig.show(save_path=save_path, verbose=verbose, scale=scale, interactive=interactive)


class ImageDisplay(Display):
    """An object to combine various image components for visualization

    Args:
        image: An image to be displayed. 3-dimensional torch tensors are generally assumed to be channel first,
            while tf and np are assumed to be channel last. Either way, only 1 or 3 channel images are supported.
        text: Text which will be printed in the center of this figure.
        masks: One or more 2-dimensional tensors. They may be combined with labels if desired (<mask>, <label>).
            Tensors may be 3-dimensional with the last dimension indicating multiple different masks.
        bboxes: One or more bounding boxes of the form (x0, y0, width, height [, label]), where (x0, y0) is the top
            left corner of the box. These may also be encoded in a tensor of shape (4,) or (N,4) for multiple boxes.
        keypoints: One or more keypoints of the form (x, y [, label]). These may also be encoded in a tensor of shape
            (2,) or (N,2) for multiple keypoints.
        title: What should the title of this figure be.
        color_map: How to color 1-channel images. Options from: https://plotly.com/python/builtin-colorscales/

    Raises:
        AssertionError: If the provided arguments violate expected type/shape constraints.
    """
    def __init__(self,
                 image: Union[None, 'Tensor'] = None,
                 text: Union[None, str, 'Tensor'] = None,
                 masks: Union[None, 'Tensor', Tuple['Tensor', str], Sequence['Tensor'],
                              Sequence[Tuple['Tensor', str]]] = None,
                 bboxes: Union[None, 'BoundingBox', 'Tensor', Sequence['BoundingBox'], Sequence['Tensor']] = None,
                 keypoints: Union[None, 'KeyPoint', 'Tensor', Sequence['KeyPoint'], Sequence['Tensor']] = None,
                 title: Union[None, str] = None,
                 color_map: str = "gray"
                 ):

        if image is not None:
            shape = image.shape
            assert len(shape) in (2, 3), f"Image must have 2 or 3 dimensions, but found {len(shape)}"
            if len(image.shape) == 3:
                if isinstance(image, torch.Tensor) and image.shape[0] in (1, 3) and image.shape[2] > 3:
                    # Move channel first to channel last
                    channels = list(range(len(image.shape)))
                    channels.append(channels.pop(0))
                    image = image.permute(*channels)

                assert image.shape[2] in (1, 3), f"Image must have either 1 or 3 channels, but found {image.shape[2]}"
                if image.shape[2] == 1:
                    # pyplot doesn't support (x,y,1) images, so convert them to (x,y)
                    image = np.reshape(image, (image.shape[0], image.shape[1]))  # This works on tf and torch tensors
            # Convert to numpy for consistency
            image = to_number(image)
        self.image = image

        if text is not None:
            # Convert to numpy for consistency
            text = to_number(text)
            assert len(text.shape) <= 1, f"A text tensor can have at most one value, but found {len(text.shape)}"
            if len(text.shape) == 1:
                text = text[0]
            text = text.item()
            if isinstance(text, bytes):
                text = text.decode('utf8')
            text = "{}".format(text)
        self.text = text

        masks = to_list(masks)
        masks = [(mask, '') if not isinstance(mask, tuple) else mask for mask in masks]
        self.masks = []
        self.n_masks = 0
        for mask_tuple in masks:
            assert isinstance(mask_tuple, tuple), \
                "Masks must be tuples of the form (<tensor>, <label>) or else simply a raw tensor"
            assert len(mask_tuple) == 2, "Masks must be tuples of the form (<tensor>, <label>)"
            assert isinstance(mask_tuple[1], str), "Masks must be tuples of the form (<tensor>, <label>)"
            mask = to_number(mask_tuple[0])
            assert len(mask.shape) in (2, 3), "Masks must be 2 dimensional, or 3 dimensional with the last " \
                                              f"dimension indicating multiple masks, but found {len(mask.shape)}"
            # Give all masks a channel dimension for consistency
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            # Move the channels to the front for easy iteration
            mask = np.moveaxis(mask, -1, 0)
            self.n_masks = max(self.n_masks, mask.shape[0])
            # Add an axis on the end which will be used for colors later
            mask = np.expand_dims(mask, -1)
            self.masks.append((mask, mask_tuple[1]))

        bboxes = to_list(bboxes)
        self.bboxes = []
        self.n_bboxes = 0
        for bbox in bboxes:
            if hasattr(bbox, 'shape'):
                assert len(bbox.shape) in (1, 2), "Bounding box tensors must be 1 dimensional, or 2 dimensional " \
                                                  "with the first dimension indicating multiple boxes, but found " \
                                                  f"{len(bbox.shape)}"
                assert bbox.shape[-1] in (4, 5), "Bounding boxes must contain either 4 or 5 elements: (x0, y0, width," \
                                                 f" height [,label]), but found {bbox.shape[-1]}"
                bbox = to_number(bbox)
                # Give all the bboxes a channel dimension for consistency
                if len(bbox.shape) == 1:
                    bbox = np.expand_dims(bbox, axis=0)
                self.n_bboxes = max(self.n_bboxes, bbox.shape[0])
            else:
                self.n_bboxes = max(self.n_bboxes, 1)
                assert len(bbox) in (4, 5), "Bounding boxes must contain either 4 or 5 elements: (x0, y0, width," \
                                            f" height [,label]), but found {len(bbox)}"
                # Add a channel dimension for consistency
                bbox = [bbox]  # TODO - non-tensor bbox should stack together to get different colors?
            self.bboxes.append(bbox)

        # TODO - keypoint handling
        self.keypoints = keypoints
        self.title = title or ''
        self.color_map = color_map

    def _make_image(self, im: np.ndarray) -> Image:
        im_max = np.max(im)
        im_min = np.min(im)
        if np.issubdtype(im.dtype, np.integer):
            # im is already in int format
            im = im.astype(np.uint8)
        elif 0 <= im_min <= im_max <= 1:  # im is [0,1]
            im = (im * 255).astype(np.uint8)
        elif -0.5 <= im_min < 0 < im_max <= 0.5:  # im is [-0.5, 0.5]
            im = ((im + 0.5) * 255).astype(np.uint8)
        elif -1 <= im_min < 0 < im_max <= 1:  # im is [-1, 1]
            im = ((im + 1) * 127.5).astype(np.uint8)
        else:  # im is in some arbitrary range, probably due to the Normalize Op
            ma = abs(np.max(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            mi = abs(np.min(im, axis=tuple([i for i in range(len(im.shape) - 1)]) if len(im.shape) > 2 else None))
            im = (((im + mi) / (ma + mi)) * 255).astype(np.uint8)
        # Convert (x,y) into (x,y,1) for consistency
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=-1)
        # Manually apply a colormap to 1-channel images
        if im.shape[2] == 1:
            im = np.array(sample_colorscale(colorscale=self.color_map,
                                            samplepoints=np.reshape(im, (-1)) / 255.0,
                                            colortype='tuple')).reshape((im.shape[0], im.shape[1], 3))
            im = np.rint(im * 255)
        return Image(z=im)

    def prepare(self,
                fig: Optional[Figure] = None,
                axis: Optional[Tuple[int, int]] = None,
                col_width: int = 280) -> FigureFE:
        if axis is None:
            axis = (1, 1)
        row, col = axis
        title_size = min(20, col_width // len(self.title or ' '))
        if fig is None:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[self.title] if self.title else None)
            if self.title:
                fig['layout']['annotations'][0]['font'] = {'size': title_size, 'family': 'monospace'}
        fig.update_layout({'plot_bgcolor': '#FFF'})
        x_axis_name = fig.get_subplot(row=row, col=col).xaxis.plotly_name
        y_axis_name = fig.get_subplot(row=row, col=col).yaxis.plotly_name
        fig['layout'][x_axis_name]['showticklabels'] = False
        fig['layout'][y_axis_name]['showticklabels'] = False
        # make y0 the top of the image instead of bottom (this is done automatically for Image)
        fig['layout'][y_axis_name]['autorange'] = 'reversed'
        x_axis_domain = f"{x_axis_name.split('axis')[0]} domain"
        y_axis_domain = f"{y_axis_name.split('axis')[0]} domain"
        # Put an invisible element on the plot to make other stuff work
        fig.add_annotation(text='', showarrow=False, row=row, col=col)

        if self.image is not None:
            im = self._make_image(im=self.image)
            fig.add_trace(im,
                          row=row,
                          col=col)

        empty_color = np.array((0.0, 0.0, 0.0, 0.0))  # RGBA
        mask_colors = get_colors(n_colors=self.n_masks, alpha=0.4, as_numbers=True)
        mask_colors = [np.array(color) for color in mask_colors]
        for mask_tuple in self.masks:
            mask, label = mask_tuple
            # Mask will be channel x width x height x 1
            for color_idx, msk in enumerate(mask):
                positive_color = mask_colors[color_idx]
                msk = np.where(msk, positive_color, empty_color)
                # TODO - handle labeling
                msk = Image(z=msk, colormodel='rgba', hoverinfo=None)
                fig.add_trace(msk, row=row, col=col)

        # # Works, and legend interactivity, but slow
        # mask_legend = defaultdict(lambda: True)
        # mask_colors = get_colors(n_colors=self.n_masks, alpha=0.3)
        # for mask_tuple in self.masks:
        #     mask, label = mask_tuple
        #     # Mask will be channel x width x height x 1
        #     for color_idx, msk in enumerate(mask):
        #         y, x = np.where(np.squeeze(msk))
        #         mask_title = label or f"{color_idx}"
        #         for y_c, x_c in zip(y, x):
        #             point = Scatter(x=[x_c-0.5, x_c+0.5, x_c+0.5, x_c-0.5],
        #                             y=[y_c+0.5, y_c+0.5, y_c-0.5, y_c-0.5],
        #                             mode='lines',
        #                             fill='toself',
        #                             fillcolor=mask_colors[color_idx],
        #                             name=mask_title,
        #                             legendgroup=mask_title,
        #                             showlegend=mask_legend[mask_title],
        #                             text=mask_title,
        #                             line={'width': 0})
        #             mask_legend[mask_title] = False
        #             fig.add_trace(point, row=row, col=col)

        bbox_colors = get_colors(n_colors=self.n_bboxes)
        for bbox_set in self.bboxes:
            for color_idx, bbox in enumerate(bbox_set):
                # Bounding Box Data. Should be (x0, y0, w, h, <label>)
                # Unpack the box, which may or may not have a label
                x0 = float(bbox[0])
                y0 = float(bbox[1])
                width = float(bbox[2])
                height = float(bbox[3])
                color = bbox_colors[color_idx]
                label = None if len(bbox) < 5 else str(bbox[4])

                # Don't draw empty boxes, or invalid box
                if width <= 0 or height <= 0:
                    continue
                fig.add_shape({'type': 'rect',
                               'x0': x0,
                               'x1': x0 + width,
                               'y0': y0,
                               'y1': y0 + height,
                               'line_color': color,
                               'line_width': 3},
                              row=row,
                              col=col,
                              exclude_empty_subplots=False)
                if label is not None:
                    font_size = max(8, min(14, int(width // len(label or ' '))))
                    # One annotation with translucent background
                    fig.add_annotation(x=x0,
                                       y=y0,
                                       xshift=len(label)*font_size/2,
                                       yshift=font_size,
                                       text=label,
                                       showarrow=False,
                                       font={'size': font_size,
                                             'color': 'white',
                                             'family': 'monospace'},
                                       bgcolor='white',
                                       opacity=0.6,
                                       exclude_empty_subplots=False,
                                       row=row,
                                       col=col)
                    # Another to make the text opaque
                    fig.add_annotation(x=x0,
                                       y=y0,
                                       xshift=len(label)*font_size/2,
                                       yshift=font_size,
                                       text=label,
                                       showarrow=False,
                                       font={'size': font_size,
                                             'color': color,
                                             'family': 'monospace'},
                                       exclude_empty_subplots=False,
                                       row=row,
                                       col=col)

        if self.text:
            fig.add_annotation(text=self.text,
                               font={'size': min(45, col_width // len(self.text or ' ')),
                                     'color': 'Black',
                                     'family': 'monospace'},
                               showarrow=False,
                               xref=x_axis_domain,
                               xanchor='center',
                               x=0.5,
                               yref=y_axis_domain,
                               yanchor='middle',
                               y=0.5,
                               exclude_empty_subplots=False,
                               row=row,
                               col=col)

        if not isinstance(fig, FigureFE):
            fig = FigureFE.from_figure(fig)

        return fig


class BatchDisplay(Display):
    """An object to combine various batched image components for visualization

    Args:
        image: A batch of image to be displayed. 4-dimensional torch tensors are generally assumed to be channel first,
            while tf and np are assumed to be channel last. Either way, only 1 or 3 channel images are supported.
        text: Text which will be printed in the center of each figure.
        masks: Batches of one or more 2-dimensional tensors. They may be combined with labels if desired:
            Bx(<mask>, <label>). Tensors may be 3-dimensional with the second dimension indicating multiple different
            masks: BxNx<mask>x<label>.
        bboxes: Batches of one or more bounding boxes of the form (x0, y0, width, height [, label]), where (x0, y0) is
            the top left corner of the box. These may also be encoded in a tensor of shape (4,) or (N,4) for multiple
            boxes.
        keypoints: Batches of one or more keypoints of the form (x, y [, label]). These may also be encoded in a tensor
            of shape (2,) or (N,2) for multiple keypoints.
        title: What should the title of this figure be.
        color_map: How to color 1-channel images. Options from: https://plotly.com/python/builtin-colorscales/

    Raises:
        AssertionError: If the provided arguments violate expected type/shape constraints.
    """
    def __init__(self,
                 image: Union[None, 'Tensor', Sequence['Tensor']] = None,
                 text: Union[None, Sequence[str], 'Tensor', Sequence['Tensor']] = None,
                 masks: Union[None, 'Tensor', Sequence[Tuple['Tensor', str]], Sequence[Sequence['Tensor']],
                              Sequence[Sequence[Tuple['Tensor', str]]]] = None,
                 bboxes: Union[None, Sequence['BoundingBox'], 'Tensor', Sequence['Tensor'],
                               Sequence[Sequence['BoundingBox']], Sequence[Sequence['Tensor']]] = None,
                 keypoints: Union[None, Sequence['KeyPoint'], 'Tensor', Sequence['Tensor'],
                                  Sequence[Sequence['KeyPoint']], Sequence[Sequence['Tensor']]] = None,
                 title: Union[None, str] = None,
                 color_map: str = "greys"
                 ):
        self.batch = []
        for img, txt, mask, bbox, keypoint in zip_longest([] if image is None else image,
                                                          [] if text is None else text,
                                                          [] if masks is None else masks,
                                                          [] if bboxes is None else bboxes,
                                                          [] if keypoints is None else keypoints,
                                                          fillvalue=None):
            self.batch.append(ImageDisplay(image=img,
                                           text=txt,
                                           masks=mask,
                                           bboxes=bbox,
                                           keypoints=keypoint,
                                           title=None,
                                           color_map=color_map))
        self.batch_size = len(self.batch) or 1
        self.title = title or ''

    def prepare(self,
                fig: Optional[Figure] = None,
                col: Optional[int] = None,
                col_width: int = 400) -> FigureFE:

        if fig is None:
            fig = make_subplots(rows=self.batch_size,
                                cols=1,
                                column_titles=[self.title],
                                vertical_spacing=0.005)
            # Update the figure title text size
            fig.for_each_annotation(lambda a: a.update(font={
                'size': min(20, 3 + int(1 + col_width) // len(a.text or ' ')),
                'family': 'monospace'
            }))
        if col is None:
            col = 1
        for row, elem in enumerate(self.batch, start=1):
            fig = elem.prepare(fig=fig, axis=(row, col), col_width=col_width)

        fig.update_layout(width=col_width,
                          height=col_width * self.batch_size,
                          margin={'l': 0, 'r': 0, 'b': 40, 't': 60}
                          )

        if not isinstance(fig, FigureFE):
            fig = FigureFE.from_figure(fig)

        return fig


class GridDisplay(Display):
    def __init__(self, columns: Sequence[Union[BatchDisplay, ImageDisplay]]):
        self.batch_size = None
        for col in columns:
            if self.batch_size is None:
                if isinstance(col, ImageDisplay):
                    self.batch_size = 1
                else:
                    self.batch_size = col.batch_size
            else:
                if isinstance(col, BatchDisplay):
                    assert self.batch_size == col.batch_size, \
                        f"Found inconsistent batch sizes: {self.batch_size} and {col.batch_size}"
                else:
                    assert self.batch_size == 1, "Cannot mix ImageDisplay and BatchDisplay within a GridDisplay"
        self.columns = columns

    def prepare(self) -> FigureFE:
        vertical_gap = 0.005
        horizontal_gap = 0.005
        n_cols = len(self.columns)

        if in_notebook():
            # Jupyter wastes some screen space, so max width needs to be smaller
            max_width = 1000
        else:
            max_width = 1500
        im_width = min(max_width, 280 * n_cols)
        im_height = int(im_width * max(self.batch_size / n_cols, 0.65))
        col_width = im_width // n_cols

        fig = make_subplots(rows=self.batch_size,
                            cols=n_cols,
                            column_titles=[col.title for col in self.columns],
                            vertical_spacing=vertical_gap,
                            horizontal_spacing=horizontal_gap)
        # Update the figure title text size
        fig.for_each_annotation(lambda a: a.update(font={
            'size': min(20, 3 + int(1 + col_width) // len(a.text or ' ')),
            'family': 'monospace'
        }))

        for col_idx, col in enumerate(self.columns, start=1):
            if isinstance(col, BatchDisplay):
                fig = col.prepare(fig=fig, col=col_idx, col_width=col_width)
            else:
                fig = col.prepare(fig=fig, axis=(1, col_idx), col_width=col_width)

        fig.update_layout(width=im_width,
                          height=im_height,
                          margin={'l': 0, 'r': 0, 'b': 40, 't': 60})

        if not isinstance(fig, FigureFE):
            fig = FigureFE.from_figure(fig)

        return fig
