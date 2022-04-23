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
import argparse
import sys
from typing import Any, Dict, List

from fastestimator.util.cli_util import SaveAction


def search(args: Dict[str, Any], unknown: List[str]) -> None:
    """A method to invoke the FE search logging function using CLI-provided arguments.

    Args:
        args: The arguments to be fed to the visualize_search() method.
        unknown: Any cli arguments not matching known inputs for the visualize_search() method.

    Raises:
        SystemExit: If `unknown` arguments were provided by the user.
    """
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    if args['draw'] == 'cartesian':
        from fastestimator.search.visualize.cartesian import visualize_cartesian
        fn = visualize_cartesian
    elif args['draw'] == 'heatmap':
        from fastestimator.search.visualize.heatmap import visualize_heatmap
        fn = visualize_heatmap
    elif args['draw'] == 'parallel':
        from fastestimator.search.visualize.parallel_coordinate_plot import visualize_parallel_coordinates
        fn = visualize_parallel_coordinates
    else:
        from fastestimator.search.visualize.visualize import visualize_search
        fn = visualize_search
    kwargs = {'search': args['search_path'],
              'title': args['title'],
              'ignore_keys': args['ignore'],
              'save_path': args['save_dir']}
    # Only add the function specific args if the user provides them to avoid clashing with manually specified plot type
    if args['color_by']:
        kwargs['color_by'] = args['color_by']
    if args['group']:
        kwargs['groups'] = args['group']
    fn(**kwargs)


def configure_plot_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add a logging parser to an existing argparser.

    Args:
        subparsers: The parser object to be appended to.
    """
    parser = subparsers.add_parser('plot',
                                   description='Generates summary graph(s) from a saved search json file',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('search_path',
                        metavar='<File Path>',
                        type=str,
                        help="The path to a json file summarizing a search object")
    parser.add_argument('--ignore',
                        metavar='I',
                        type=str,
                        nargs='+',
                        help="The names of parameters or results to ignore when visualizing")
    parser.add_argument('--title',
                        metavar='T',
                        type=str,
                        help="A custom title for the generated plot",
                        default=None)
    parser.add_argument('--draw',
                        metavar='D',
                        choices=['cartesian', 'heatmap', 'parallel'],
                        help="Force the system to attempt to draw a particular type of plot. This may raise an error if"
                             " the given search instance is incompatible with the desired type of visualization. "
                             "Choices are 'cartesian', 'heatmap', or 'parallel'.",
                        default=None)
    parser.add_argument('--color_by',
                        metavar='C',
                        type=str,
                        help="Override the key used to color parallel coordinate plots",
                        default=None)
    parser.add_argument('-G',
                        '--group',
                        action='append',
                        type=str,
                        nargs='+',
                        help="Group multiple results onto the same cartesian plot")
    save_group = parser.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        dest='save',
        action=SaveAction,
        default=False,
        help="Save the output image. May be accompanied by a directory into \
                                              which the file is saved. If no output directory is specified, the log \
                                              directory will be used")
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.set_defaults(func=search)
