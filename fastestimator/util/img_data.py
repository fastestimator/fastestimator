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
from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from plotly.colors import sample_colorscale
from plotly.graph_objects import Figure, Image, Scatter
from plotly.subplots import make_subplots

from fastestimator.util.base_util import FigureFE, in_notebook, to_list
from fastestimator.util.util import to_number

if TYPE_CHECKING:
    import tensorflow as tf

    Tensor = TypeVar('Tensor', np.ndarray, tf.Tensor, torch.Tensor)
    BoundingBox = TypeVar('BoundingBox',
                          Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]],
                          Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], str])


class Display(ABC):
    @abstractmethod
    def prepare(self, **kwargs) -> FigureFE:
        raise NotImplementedError()

    def show(self, save_path: Optional[str] = None, verbose: bool = True, scale: int = 1,
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
        masks: A 2D tensor representing a mask, or a 3D tensor indicating multiple 2D masks. They may be combined with
            label(s) if desired (<mask>, <label>).
        bboxes: One or more bounding boxes of the form (x0, y0, width, height [, label]), where (x0, y0) is the top
            left corner of the box. These may also be encoded in a tensor of shape (4,) or (N,4) for multiple boxes.
        keypoints: A 1D tensor representing a keypoint of shape (2,), or a 2D tensor of shape (N,2) indicating multiple
            1D keypoints. They may be combined with label(s) if desired: (<keypoint>, <label>). (x,y) coordinates
            indicate distance from the top left of an image, with negative or None values indicating that a keypoint
            was not detected.
        title: What should the title of this figure be.
        mask_threshold: If provided, any masks will be binarized based on the given threshold value (1 if > t, else 0).
        color_map: How to color 1-channel images. Options from: https://plotly.com/python/builtin-colorscales/. If 2
            strings are provided, the first will be used to color grey-scale images and the second will be used to color
            continuous (non-thresholded) masks. If a single string is provided it will be used for both image and masks.

    Raises:
        AssertionError: If the provided arguments violate expected type/shape constraints.
    """
    def __init__(self,
                 image: Union[None, 'Tensor'] = None,
                 text: Union[None, str, 'Tensor'] = None,
                 masks: Union[None, 'Tensor', Tuple['Tensor', Union[str, Sequence[str]]]] = None,
                 bboxes: Union[None, 'Tensor', Sequence['BoundingBox'], Sequence['Tensor']] = None,
                 keypoints: Union[None, 'Tensor', Tuple['Tensor', Union[str, Sequence[str]]]] = None,
                 title: Union[None, str] = None,
                 mask_threshold: Optional[float] = None,
                 color_map: Union[str, Tuple[str, str]] = ("gray", "turbo")):

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

        masks, mask_labels, *_ = to_list(masks) + [None, None]
        mask_labels = to_list(mask_labels)
        if masks is not None:
            if isinstance(masks, torch.Tensor) and len(masks.shape) == 3 and self.image is not None:
                # Unfortunately we can't just permute all torch tensors since TF users also might have torch tensors if
                #  they were using pipeline.get_results(). If there's an accompanying image though we can figure it out
                if masks.shape[1] == self.image.shape[0] and masks.shape[2] == self.image.shape[1]:
                    # Move channel first to channel last
                    masks = masks.permute(1, 2, 0)
            masks = to_number(masks)
            assert len(masks.shape) in (2, 3), "Masks must be 2 dimensional, or 3 dimensional with the last " \
                                               f"dimension (tf/np) or first dimension (torch) indicating multiple " \
                                               f"masks, but found {len(masks.shape)}"
            # Give all masks a channel dimension for consistency
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=-1)
            # Move the channels to the front for easy iteration
            masks = np.moveaxis(masks, -1, 0)
            if mask_threshold is not None:
                masks = np.where(masks > mask_threshold, 1.0, 0.0)
            # If there are multiple continuous probability masks, compress them into a single mask since overlaying
            # solid color patches over the entire image is pointless
            if np.unique(masks).size > 2:
                masks = np.max(masks, axis=0, keepdims=True)
                assert len(mask_labels) == 0, "When probabilistic masks are provided and no mask_threshold is set, " \
                                              "then mask labels cannot be used"
            # Add an axis on the end which will be used for colors later
            masks = np.expand_dims(masks, -1)
            # Ensure mask labels are sufficient if provided
            if mask_labels:
                assert len(mask_labels) == len(masks), "If mask labels are provided they must be 1-1 with the number " \
                                                       f"of masks, but found {len(mask_labels)} labels and " \
                                                       f"{len(masks)} masks"
            else:
                # Make default blank labels for easy zip later
                mask_labels = [''] * len(masks)
        self.masks = [] if masks is None else masks
        self.mask_labels = [] if mask_labels is None else mask_labels

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

        keypoints, keypoint_labels, *_ = to_list(keypoints) + [None, None]
        keypoint_labels = to_list(keypoint_labels)
        if keypoints is not None:
            keypoints = to_number(keypoints)
            assert len(keypoints.shape) in (1, 2), "Keypoints must be 1 dimensional, or 2 dimensional with the first " \
                                                   "dimension indicating multiple keypoints, but found " \
                                                   f"{len(keypoints.shape)}"
            # Give all keypoints a channel dimension for consistency
            if len(keypoints.shape) == 1:
                keypoints = np.expand_dims(keypoints, axis=0)

            assert keypoints.shape[1] == 2, "Keypoints should contain 2 coordinates (x,y) but found " \
                                            f"{keypoints.shape[1]}"
            # Ensure keypoint labels are sufficient if provided
            if keypoint_labels:
                assert len(keypoint_labels) == len(keypoints), "If keypoint labels are provided they must be 1-1 " \
                                                               "with the number of keypoints, but found " \
                                                               f"{len(keypoint_labels)} labels and {len(keypoints)} " \
                                                               f"keypoints"
            else:
                # Make default blank labels for easy zip later
                keypoint_labels = [''] * len(keypoints)
        self.keypoints = [] if keypoints is None else keypoints
        self.keypoint_labels = [] if keypoint_labels is None else keypoint_labels

        self.title = title or ''
        color_map = to_list(color_map)
        self.img_color_map = color_map[0]
        self.mask_color_map = color_map[-1]

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
            im = np.array(
                sample_colorscale(colorscale=self.img_color_map,
                                  samplepoints=np.reshape(im, (-1)) / 255.0,
                                  colortype='tuple')).reshape((im.shape[0], im.shape[1], 3))
            im = np.rint(im * 255)
        return Image(z=im)

    def prepare(self, fig: Optional[Figure] = None, axis: Optional[Tuple[int, int]] = None,
                col_width: int = 280) -> FigureFE:
        if axis is None:
            axis = (1, 1)
        row, col = axis
        title_size = min(20, col_width // len(self.title or ' '))
        if fig is None:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[self.title] if self.title else None)
            if self.title:
                fig['layout']['annotations'][0]['font'] = {'size': title_size, 'family': 'monospace'}

        if not isinstance(fig, FigureFE):
            fig = FigureFE.from_figure(fig)

        fig.update_layout({'plot_bgcolor': '#FFF'})
        fig.update_layout(legend=dict(itemsizing='constant'))  # Make lines in legend thicker for masks
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
            fig.add_trace(im, row=row, col=col)

        empty_color = np.array((0.0, 0.0, 0.0, 0.0))  # RGBA
        for color_idx, (mask, label) in enumerate(zip(self.masks, self.mask_labels)):
            # Mask will be width x height x 1
            if np.max(mask) > 1:
                mask = mask / 255.0
            if len(self.masks) == 1 and np.unique(mask).size > 2:
                # continuous heatmap
                msk_heatmap = np.array(
                    sample_colorscale(colorscale=self.mask_color_map,
                                      samplepoints=np.reshape(mask, (-1)),
                                      colortype='tuple')).reshape((mask.shape[0], mask.shape[1], 3))
                msk_heatmap = np.rint(msk_heatmap * 255)
                # add alpha channel
                mask = np.concatenate([msk_heatmap, 0.5 * np.ones_like(mask)], axis=-1)
                fig.add_trace(Image(z=mask, colormodel='rgba', hoverinfo=None),
                              row=row,
                              col=col,
                              exclude_empty_subplots=False)
            else:
                if label:
                    # Draw the mask out dynamically to allow legend interactivity
                    for row_idx, mask_row in enumerate(np.squeeze(mask)):
                        start = 0
                        active = False
                        for col_idx, mask_col in enumerate(mask_row):
                            if mask_col == 1:
                                if not active:
                                    active = True
                                    start = col_idx
                            else:
                                if active:
                                    # Only reserve a color once we know for sure that we're active to avoid a batch
                                    # situation where no legend is displayed
                                    color, show = fig._get_color(clazz='mask', label=label, n_colors=len(self.masks))
                                    line = Scatter(x=[start, col_idx],
                                                   y=[row_idx, row_idx],
                                                   mode='lines',
                                                   line={'width': 1,
                                                         'color': color},
                                                   name=label,
                                                   legendgroup=f"mask_{label}",
                                                   legendrank=0,  # Sort mask labels higher than bbox and keypoint
                                                   showlegend=show,
                                                   text=label)
                                    active = False
                                    fig.add_trace(line, row=row, col=col, exclude_empty_subplots=False)
                        if active:
                            # The mask went all the way to the right side of the image
                            color, show = fig._get_color(clazz='mask', label=label, n_colors=len(self.masks))
                            line = Scatter(x=[start, len(mask_row) - 1],
                                           y=[row_idx, row_idx],
                                           mode='lines',
                                           line={'width': 1,
                                                 'color': color},
                                           name=label,
                                           legendgroup=f"mask_{label}",
                                           legendrank=0,
                                           showlegend=show,
                                           text=label)
                            fig.add_trace(line, row=row, col=col, exclude_empty_subplots=False)
                else:
                    # Draw the mask as a static image for efficiency
                    positive_color, _ = fig._get_color(clazz='mask',
                                                       label=label or color_idx,
                                                       n_colors=len(self.masks),
                                                       as_numbers=True)
                    mask = np.where(mask, np.array(positive_color), empty_color)
                    fig.add_trace(Image(z=mask, colormodel='rgba', hoverinfo=None),
                                  row=row,
                                  col=col,
                                  exclude_empty_subplots=False)

        for bbox_set in self.bboxes:
            for color_idx, bbox in enumerate(bbox_set):
                # Bounding Box Data. Should be (x0, y0, w, h, <label>)
                # Unpack the box, which may or may not have a label
                x0 = float(bbox[0])
                y0 = float(bbox[1])
                width = float(bbox[2])
                height = float(bbox[3])
                label = None if len(bbox) < 5 else str(bbox[4])
                color, show = fig._get_color(clazz='bbox', label=label or color_idx, n_colors=self.n_bboxes)

                # Don't draw empty boxes, or invalid box
                if width <= 0 or height <= 0:
                    continue
                kwargs = {'x': [x0, x0, x0+width, x0+width, x0],
                          'y': [y0, y0+height, y0+height, y0, y0],
                          'mode': 'lines',
                          'line': {'color': color, 'width': 3},
                          'showlegend': False,
                          }
                if label:
                    kwargs['name'] = label
                    kwargs['legendrank'] = 1
                    kwargs['legendgroup'] = f'bb_{label}'
                    kwargs['showlegend'] = show
                    # Add label text onto image
                    font_size = max(8, min(14, int(width // len(label or ' '))))
                    fig.add_trace(Scatter(x=[x0],
                                          y=[y0-1],  # A slight offset to help text not run in to bbox
                                          mode='text',
                                          text='<span style="text-shadow: -1px 1px 0 #FFFFFF, '
                                               '1px 1px 0px #FFFFFF, 1px -1px 0px #FFFFFF, -1px -1px 0px #FFFFFF;">'
                                               f'{label}</span>',
                                          textposition="top right",
                                          legendgroup=f'bb_{label}',
                                          hoverinfo='skip',
                                          textfont={'size': font_size,
                                                    'color': color,
                                                    'family': 'monospace'},
                                          showlegend=False,
                                          legendrank=1),
                                  row=row,
                                  col=col,
                                  exclude_empty_subplots=False)
                fig.add_trace(Scatter(**kwargs),
                              row=row,
                              col=col,
                              exclude_empty_subplots=False)

        for keypoint, label in zip(self.keypoints, self.keypoint_labels):
            x, y = keypoint
            if (x is None) or (x < 0) or (y is None) or (y < 0):
                # Skip negative or None key-points
                continue
            kwargs = {'x': [x],
                      'y': [y],
                      'mode': 'markers',
                      'showlegend': False,
                      'marker': {'color': 'red',
                                 'size': 10,
                                 'symbol': 'circle'}}
            if label:
                kwargs['name'] = label
                kwargs['legendgroup'] = f"keypoint_{label}"
                color, show = fig._get_color(clazz='keypoints', label=label, n_colors=len(self.keypoints))
                kwargs['showlegend'] = show
                kwargs['marker']['color'] = color
            fig.add_trace(Scatter(**kwargs), row=row, col=col, exclude_empty_subplots=False)

        if self.text:
            fig.add_annotation(
                text=self.text,
                font={
                    'size': min(45, col_width // len(self.text or ' ')), 'color': 'Black', 'family': 'monospace'
                },
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

        return fig


class BatchDisplay(Display):
    """An object to combine various batched image components for visualization

    Args:
        image: A batch of image to be displayed. 4-dimensional torch tensors are generally assumed to be channel first,
            while tf and np are assumed to be channel last. Either way, only 1 or 3 channel images are supported.
        text: Text which will be printed in the center of each figure.
        masks: A 3D tensor representing a batch of 2D masks, or a 4D tensor indicating a batch of 2D masks having
            multiple channels. They may be combined with label(s) if desired (<mask>, <label>). For masks with C
            channels, C labels should be provided (which will then be used for every element in the batch).
        bboxes: Batches of one or more bounding boxes of the form (x0, y0, width, height [, label]), where (x0, y0) is
            the top left corner of the box. These may also be encoded in a tensor of shape (4,) or (N,4) for multiple
            boxes.
        keypoints: A 2D tensor representing a batch of 1D keypoints, or a 3D tensor indicating a batch of sets of
            1D keypoints. They may be combined with label(s) if desired: (<keypoint>, <label>). For a batch N with C
            keypoints per element, C labels should be provided (which will then be used for every element in the batch).
            The keypoint shape should be (N, C, 2) or (N, 2). (x,y) coordinates indicate distance from the top left of
            an image, with negative or None values indicating that a keypoint was not detected.
        title: What should the title of this figure be.
        mask_threshold: If provided, any masks will be binarized based on the given threshold value (1 if > t, else 0).
        color_map: How to color 1-channel images. Options from: https://plotly.com/python/builtin-colorscales/. If 2
            strings are provided, the first will be used to color grey-scale images and the second will be used to color
            continuous (non-thresholded) masks. If a single string is provided it will be used for both image and masks.

    Raises:
        AssertionError: If the provided arguments violate expected type/shape constraints.
    """
    def __init__(self,
                 image: Union[None, 'Tensor', Sequence['Tensor']] = None,
                 text: Union[None, Sequence[str], 'Tensor', Sequence['Tensor']] = None,
                 masks: Union[None, 'Tensor', Tuple['Tensor', Union[str, Sequence[str]]]] = None,
                 bboxes: Union[None,
                               'Tensor',
                               Sequence['Tensor'],
                               Sequence[Sequence['BoundingBox']],
                               Sequence[Sequence['Tensor']]] = None,
                 keypoints: Union[None, 'Tensor', Tuple['Tensor', Union[str, Sequence[str]]]] = None,
                 title: Union[None, str] = None,
                 mask_threshold: Optional[float] = None,
                 color_map: Union[str, Tuple[str, str]] = ("gray", "turbo")):
        self.batch = []
        masks, mask_labels, *_ = to_list(masks) + [None, None]
        keypoints, keypoint_labels, *_ = to_list(keypoints) + [None, None]
        mask_labels = to_list(mask_labels)
        keypoint_labels = to_list(keypoint_labels)
        for img, txt, mask, bbox, keypoint in zip_longest([] if image is None else image,
                                                          [] if text is None else text,
                                                          [] if masks is None else masks,
                                                          [] if bboxes is None else bboxes,
                                                          [] if keypoints is None else keypoints,
                                                          fillvalue=None):
            self.batch.append(
                ImageDisplay(image=img,
                             text=txt,
                             masks=(mask, mask_labels) if mask_labels else mask,
                             bboxes=bbox,
                             keypoints=(keypoint, keypoint_labels) if keypoint_labels else keypoint,
                             title=None,
                             mask_threshold=mask_threshold,
                             color_map=color_map))
        self.batch_size = len(self.batch) or 1
        self.title = title or ''

    def prepare(self, fig: Optional[Figure] = None, col: Optional[int] = None, col_width: int = 400) -> FigureFE:

        if fig is None:
            fig = make_subplots(rows=self.batch_size, cols=1, column_titles=[self.title], vertical_spacing=0.005)
            # Update the figure title text size
            fig.for_each_annotation(
                lambda a: a.update(font={
                    'size': min(20, 3 + int(1 + col_width) // len(a.text or ' ')), 'family': 'monospace'
                }))
        if col is None:
            col = 1
        for row, elem in enumerate(self.batch, start=1):
            fig = elem.prepare(fig=fig, axis=(row, col), col_width=col_width)

        fig.update_layout(width=col_width,
                          height=col_width * self.batch_size,
                          margin={
                              'l': 0, 'r': 0, 'b': 40, 't': 60
                          })

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
            'size': min(20, 3 + int(1 + col_width) // len(a.text or ' ')), 'family': 'monospace'
        }))

        for col_idx, col in enumerate(self.columns, start=1):
            if isinstance(col, BatchDisplay):
                fig = col.prepare(fig=fig, col=col_idx, col_width=col_width)
            else:
                fig = col.prepare(fig=fig, axis=(1, col_idx), col_width=col_width)

        fig.update_layout(width=im_width, height=im_height, margin={'l': 0, 'r': 0, 'b': 40, 't': 60})

        if not isinstance(fig, FigureFE):
            fig = FigureFE.from_figure(fig)

        return fig
