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
import matplotlib.pyplot as plt
import numpy as np


def show_text(background, text, axis=None, title=None):
    """Plots a given image onto an axis

    Args:
        axis: The matplotlib axis to plot on
        background: A background image to display behind the text (useful for sizing the plot correctly)
        text: The text to display
        title: A title for the image
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    # matplotlib doesn't support (x,y,1) images, so convert them to (x,y)
    if len(background.shape) == 3 and background.shape[2] == 1:
        background = np.reshape(background, (background.shape[0], background.shape[1]))

    axis.axis('off')
    axis.imshow(background, cmap=plt.get_cmap(name="Greys_r"), vmin=0, vmax=1)
    axis.text(0.5, 0.5, text, ha='center', va='center', transform=axis.transAxes, wrap=False, family='monospace')
    if title is not None:
        axis.set_title(title)


def show_image(im, axis=None, title=None):
    """Plots a given image onto an axis

    Args:
        axis: The matplotlib axis to plot on, or None for a new plot
        im: The image to display (width X height)
        title: A title for the image
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    axis.axis('off')
    im = np.asarray(im)
    if np.max(im) <= 1 and np.min(im) >= 0:  # im is [0,1]
        im = (im * 255).astype(np.uint8)
    elif np.max(im) <= 1:  # im is [-1, 1]
        im = ((im + 1) * 127.5).astype(np.uint8)
    else:  # im is already 255
        im = im.astype(np.uint8)
    # matplotlib doesn't support (x,y,1) images, so convert them to (x,y)
    if len(im.shape) == 3 and im.shape[2] == 1:
        im = np.reshape(im, (im.shape[0], im.shape[1]))
    axis.imshow(im)
    if title is not None:
        axis.set_title(title)


def show_gray_image(im, axis=None, title=None, color_map="inferno"):
    """Plots a given image onto an axis

    Args:
        axis: The matplotlib axis to plot on, or None for a new plot
        im: The image to display (width X height)
        title: A title for the image
        color_map: The color set to be used (since the image is gray scale)
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    axis.axis('off')
    axis.imshow(im, cmap=plt.get_cmap(name=color_map), vmin=0, vmax=1)
    if title is not None:
        axis.set_title(title)


def fig_to_img(fig, batch=True):
    flat_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    flat_image_pixels = flat_image.shape[0] // 3
    width, height = fig.canvas.get_width_height()
    if flat_image_pixels % height != 0:
        # Canvas returned incorrect width/height. This seems to happen sometimes in Jupyter. TODO: figure out why.
        search = 1
        guess = height + search
        while flat_image_pixels % guess != 0:
            if search < 0:
                search = -1 * search + 1
            else:
                search = -1 * search
            guess = height + search
        height = guess
        width = flat_image_pixels // height
    shape = (1, height, width, 3) if batch else (height, width, 3)
    return flat_image.reshape(shape)
