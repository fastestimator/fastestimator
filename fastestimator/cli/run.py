# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import json
import os
import sys
from typing import Any, Dict, List, Optional

from fastestimator.util.cli_util import parse_cli_to_dictionary


def run(args: Dict[str, Any], unknown: Optional[List[str]]) -> None:
    """Invoke the fastestimator_run function from a file.

    Args:
        args: A dictionary containing location of the FE file under the 'entry_point' key, as well as an optional
            'hyperparameters_json' key if the user is storing their parameters in a file.
        unknown: The remainder of the command line arguments to be passed along to the fastestimator_run() method.
    """
    entry_point = args['entry_point']
    hyperparameters = {}
    if args['hyperparameters_json']:
        hyperparameters = os.path.abspath(args['hyperparameters_json'])
        with open(hyperparameters, 'r') as f:
            hyperparameters = json.load(f)
    hyperparameters.update(parse_cli_to_dictionary(unknown))
    module_name = os.path.splitext(os.path.basename(entry_point))[0]
    dir_name = os.path.abspath(os.path.dirname(entry_point))
    sys.path.insert(0, dir_name)
    spec_module = __import__(module_name, globals(), locals())
    if hasattr(spec_module, "fastestimator_run"):
        spec_module.fastestimator_run(**hyperparameters)
    elif hasattr(spec_module, "get_estimator"):
        est = spec_module.get_estimator(**hyperparameters)
        if "train" in est.pipeline.data:
            est.fit()
        if "test" in est.pipeline.data:
            est.test()
    else:
        raise ValueError("The file {} does not contain 'fastestimator_run' or 'get_estimator'".format(module_name))


def configure_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add a run parser to an existing argparser.

    Args:
        subparsers: The parser object to be appended to.
    """
    parser = subparsers.add_parser('run',
                                   description='Execute fastestimator_run function',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    # use an argument group for required flag arguments since otherwise they will show up as optional in the help
    parser.add_argument('entry_point', type=str, help='The path to the python file')
    parser.add_argument('--hyperparameters',
                        dest='hyperparameters_json',
                        type=str,
                        help="The path to the hyperparameters JSON file")
    parser.add_argument_group(
        'hyperparameter arguments',
        'Arguments to be passed through to the fastestimator_run() call. \
        Examples might look like --epochs <int>, --batch_size <int>, --optimizer <str>, etc...')
    parser.set_defaults(func=run)
