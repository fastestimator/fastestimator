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
import argparse
import sys
from typing import Any, Dict, List

from fastestimator.cli.cli_util import SaveAction
from fastestimator.summary.logs import parse_log_dir


def logs(args: Dict[str, Any], unknown: List[str]) -> None:
    """A method to invoke the FE logging function using CLI-provided arguments.

    Args:
        args: The arguments to be fed to the parse_log_dir() method.
        unknown: Any cli arguments not matching known inputs for the parse_log_dir() method.

    Raises:
        SystemExit: If `unknown` arguments were provided by the user.
    """
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    group_by = args['group_by']
    if isinstance(group_by, list):
        group_by = group_by[0]
    if group_by == '_n':
        group_by = r'(.*)_[\d]+' + '\\' + args['extension']
    parse_log_dir(args['log_dir'],
                  args['extension'],
                  args['recursive'],
                  args['smooth'],
                  args['save'],
                  args['save_dir'],
                  args['ignore'],
                  args['include'],
                  args['share_legend'],
                  args['pretty_names'],
                  group_by)


def configure_log_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add a logging parser to an existing argparser.

    Args:
        subparsers: The parser object to be appended to.
    """
    parser = subparsers.add_parser('logs',
                                   description='Generates comparison graphs amongst one or more log files',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('log_dir',
                        metavar='<Log Dir>',
                        type=str,
                        help="The path to a folder containing one or more log files")
    parser.add_argument('--extension',
                        metavar='E',
                        type=str,
                        help="The file type / extension of your logs",
                        default=".txt")
    parser.add_argument('--recursive', action='store_true', help="Recursively search sub-directories for log files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ignore',
                       metavar='I',
                       type=str,
                       nargs='+',
                       help="The names of metrics to ignore though they may be present in the log files")
    group.add_argument('--include',
                       metavar='Y',
                       type=str,
                       nargs='+',
                       help="The names of metrics to include. If provided, any other metrics will be ignored.")
    parser.add_argument('--smooth',
                        metavar='<float>',
                        type=float,
                        help="The amount of gaussian smoothing to apply (zero for no smoothing)",
                        default=1)
    parser.add_argument('--pretty_names', help="Clean up the metric names for display", action='store_true')
    parser.add_argument('--group_by',
                        metavar='G',
                        type=str,
                        nargs=1,
                        help="A regex pattern to group different logs together and display their mean+-stdev. For "
                             r"example, you could use --G '(.*)_[\d]+\.txt' to group files of the form "
                             "<name>_<number>.txt by their <name>. We anticipate this being the common usecase, so you "
                             "can use --G _n as a shortcut for that functionality.")

    legend_group = parser.add_argument_group('legend arguments')
    legend_x_group = legend_group.add_mutually_exclusive_group(required=False)
    legend_x_group.add_argument('--common_legend',
                                dest='share_legend',
                                help="Generate one legend total",
                                action='store_true',
                                default=True)
    legend_x_group.add_argument('--split_legend',
                                dest='share_legend',
                                help="Generate one legend per graph",
                                action='store_false',
                                default=False)

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
    parser.set_defaults(func=logs)
