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
import sys
from typing import Callable

import wget


def bar_custom(current: float, total: float, width: int = 80) -> str:
    """Return progress bar string for given values in one of three styles depending on available width.

    This function was modified from wget source code at https://bitbucket.org/techtonik/python-wget/src/default/.

    The bar will be one of the following formats depending on available width:
        [..  ] downloaded / total
        downloaded / total
        [.. ]

    If total width is unknown or <= 0, the bar will show a bytes counter using two adaptive styles:
        %s / unknown
        %s

    If there is not enough space on the screen, do not display anything. The returned string doesn't include control
    characters like \r used to place cursor at the beginning of the line to erase previous content.

    This function leaves one free character at the end of the string to avoid automatic linefeed on Windows.

    ```python
    wget.download('http://url.com', '/save/dir', bar=fe.util.bar_custom)
    ```

    Args:
        current: The current amount of progress.
        total: The total amount of progress required by the task.
        width: The available width.

    Returns:
        A formatted string to display the current progress.
    """
    # process special case when total size is unknown and return immediately
    if not total or total < 0:
        msg = "{} / unknown".format(current)
        if len(msg) < width:  # leaves one character to avoid linefeed
            return msg
        if len("{}".format(current)) < width:
            return "{}".format(current)

    # --- adaptive layout algorithm ---
    #
    # [x] describe the format of the progress bar
    # [x] describe min width for each data field
    # [x] set priorities for each element
    # [x] select elements to be shown
    #   [x] choose top priority element min_width < avail_width
    #   [x] lessen avail_width by value if min_width
    #   [x] exclude element from priority list and repeat

    #  10% [.. ]  10/100
    # pppp bbbbb sssssss

    min_width = {
        'percent': 4,  # 100%
        'bar': 3,  # [.]
        'size': len("{}".format(total)) * 2 + 3,  # 'xxxx / yyyy'
    }
    priority = ['percent', 'bar', 'size']

    # select elements to show
    selected = []
    avail = width
    for field in priority:
        if min_width[field] < avail:
            selected.append(field)
            avail -= min_width[field] + 1  # +1 is for separator or for reserved space at
            # the end of line to avoid linefeed on Windows

    # render
    output = ''
    for field in selected:

        if field == 'percent':
            # fixed size width for percentage
            output += "{}%".format(100 * current // total).rjust(min_width['percent'])
        elif field == 'bar':  # [. ]
            # bar takes its min width + all available space
            output += wget.bar_thermometer(current, total, min_width['bar'] + avail)
        elif field == 'size':
            # size field has a constant width (min == max)
            output += "{:.2f} / {:.2f} MB".format(current / 1e6, total / 1e6).rjust(min_width['size'])

        selected = selected[1:]
        if selected:
            output += ' '  # add field separator

    return output


def callback_progress(blocks: int, block_size: int, total_size: int, bar_function: Callable[[int, int, int],
                                                                                            str]) -> None:
    """Callback function for urlretrieve that is called when a connection is created and then once for each block.

    Draws adaptive progress bar in terminal/console.

    Use sys.stdout.write() instead of "print", because it allows one more symbols at the line end without triggering a
    linefeed on Windows.

    ```python
    import wget
    wget.callback_progress = fe.util.callback_progress
    wget.download('http://url.com', '/save/dir', bar=fe.util.bar_custom)
    ```

    Args:
        blocks: number of blocks transferred so far.
        block_size: in bytes.
        total_size: in bytes, can be -1 if server doesn't return it.
        bar_function: another callback function to visualize progress.
    """
    width = min(100, wget.get_console_width())
    if width == 0:  # sys.stdout.fileno() in get_console_width() is not supported in jupyter notebook
        width = 80

    current_size = min(blocks * block_size, total_size)

    progress = bar_function(current_size, total_size, width)
    if progress:
        sys.stdout.write("\r{}".format(progress))
        if current_size >= total_size:
            sys.stdout.write("\n")
