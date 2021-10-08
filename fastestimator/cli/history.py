#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import argparse
import sys
from typing import Any, Dict, List

from fastestimator.summary.history import HistoryReader, delete, update_settings


def history_basic(args: Dict[str, Any], unknown: List[str]) -> None:
    """A method to query FE history using CLI-provided arguments.

    Args:
        args: The arguments to be fed to the read_sql() method.
        unknown: Any cli arguments not matching known inputs for the read_sql() method.

    Raises:
        SystemExit: If `unknown` arguments were provided by the user.
    """
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    with HistoryReader() as reader:
        reader.read_basic(limit=args['limit'],
                          interactive=args['interactive'],
                          include_args=args['include_args'],
                          errors=args['errors'],
                          include_pk=args['include_pk'],
                          include_traces=args['include_traces'],
                          include_features=args['include_features'],
                          include_datasets=args['include_datasets'],
                          include_pipeline=args['include_pipeline'],
                          include_network=args['include_network'],
                          as_csv=args['csv'])


def history_sql(args: Dict[str, Any], unknown: List[str]) -> None:
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    with HistoryReader() as reader:
        reader.read_sql(query=args['query'], as_csv=args['csv'], interactive=args['interactive'])


def clear_history(args: Dict[str, Any], unknown: List[str]) -> None:
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    delete(n_keep=args['retain'])


def settings(args: Dict[str, Any], unknown: List[str]) -> None:
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    update_settings(n_keep=args['keep'], n_keep_logs=args['keep_logs'])


def configure_history_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add a history parser to an existing argparser.

    Args:
        subparsers: The parser object to be appended to.
    """
    parser = subparsers.add_parser('history',
                                   description='View prior FE training histories',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('--limit', metavar='L', type=int, help="How many entries to return", default=15)
    parser.add_argument('--interactive',
                        dest='interactive',
                        help="Whether to run an interactive session which lets you look up detailed information",
                        action='store_true',
                        default=False)
    parser.add_argument('--args',
                        dest='include_args',
                        help="Whether to return a list of the args used to invoke the training",
                        action='store_true',
                        default=False)
    parser.add_argument('--errors',
                        dest='errors',
                        help="Whether to focus on failed trainings and include extra error information",
                        action='store_true',
                        default=False)
    parser.add_argument('--features',
                        dest='include_features',
                        help="Whether to return a list of the FE features used during each training",
                        action='store_true',
                        default=False)
    parser.add_argument('--datasets',
                        dest='include_datasets',
                        help="Whether to return a list of the datasets used during each training",
                        action='store_true',
                        default=False)
    parser.add_argument('--pipeline',
                        dest='include_pipeline',
                        help="Whether to return a list of the pipeline ops used during each training",
                        action='store_true',
                        default=False)
    parser.add_argument('--network',
                        dest='include_network',
                        help="Whether to return a list of the network ops used during each training",
                        action='store_true',
                        default=False)
    parser.add_argument('--traces',
                        dest='include_traces',
                        help="Whether to return a list of the traces used during each training",
                        action='store_true',
                        default=False)
    parser.add_argument('--pks',
                        dest='include_pk',
                        help="Whether to return the database primary keys of each entry",
                        action='store_true',
                        default=False)
    parser.add_argument('--csv',
                        dest='csv',
                        help='Print the response as a csv rather than a formatted table',
                        action='store_true',
                        default=False)
    sp = parser.add_subparsers()
    sql_parser = sp.add_parser('sql',
                               description='Perform a raw SQL query against the history database',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                               allow_abbrev=False)
    sql_parser.add_argument('query',
                            metavar='<Query>',
                            type=str,
                            help="ex: fastestimator history sql 'SELECT * FROM history'")
    sql_parser.add_argument('--interactive',
                            dest='interactive',
                            help="Whether to run an interactive session which lets you look up detailed information",
                            action='store_true',
                            default=False)
    sql_parser.add_argument('--csv',
                            dest='csv',
                            help='Print the response as a csv rather than a formatted table',
                            action='store_true',
                            default=False)
    sql_parser.set_defaults(func=history_sql)
    clear_parser = sp.add_parser('clear',
                                 description='Clear out old history entries to save space',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 allow_abbrev=False)
    clear_parser.add_argument('retain',
                              metavar='N',
                              nargs='?',
                              type=int,
                              help="How many of the most recent entries to keep",
                              default=20)
    clear_parser.set_defaults(func=clear_history)
    settings_parser = sp.add_parser('settings',
                                    description="Modify history settings, such as how many logs to retain",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    allow_abbrev=False)
    settings_parser.add_argument('--keep',
                                 metavar='K',
                                 type=int,
                                 help="How many of the most recent history entries to keep")
    settings_parser.add_argument('--keep_logs',
                                 metavar='L',
                                 type=int,
                                 help="How many of the most recent log entries to keep")
    settings_parser.set_defaults(func=settings)
    parser.set_defaults(func=history_basic)
