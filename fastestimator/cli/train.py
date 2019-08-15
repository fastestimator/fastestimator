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
import json
import os
import sys

from fastestimator.util.util import parse_cli_to_dictionary, get_gpu_count


def train(args, unknown):
    num_process = args['num_process']
    entry_point = args['entry_point']
    inputs_path = args['inputs']

    hyperparameters = {}
    if args['hyperparameters_json']:
        hyperparameters = os.path.abspath(args['hyperparameters_json'])
        hyperparameters = json.load(open(hyperparameters, 'r'))

    hyperparameters.update(parse_cli_to_dictionary(unknown))

    module_name = os.path.splitext(os.path.basename(entry_point))[0]
    dir_name = os.path.abspath(os.path.dirname(entry_point))
    sys.path.insert(0, dir_name)
    spec_module = __import__(module_name, globals(), locals(), ["get_estimator"])
    estimator = spec_module.get_estimator(**hyperparameters)
    estimator.num_process = num_process

    # Wait until now to spawn multiple threads so that all of the inputs are validated to be correct.
    # That way the user won't get four different copies of the same error message if they enter something wrong
    if num_process > 1 and not args['worker']:
        cmd = "mpirun -np %d -H localhost:%d --allow-run-as-root " % (num_process, num_process) + \
              "fastestimator --worker " + str.join(" ", sys.argv[1:])
        os.system(cmd)
    else:
        estimator.fit(inputs=inputs_path)


def configure_train_parser(subparsers):
    parser = subparsers.add_parser('train', description='Train a FastEstimator model',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    # use an argument group for required flag arguments since otherwise they will show up as optional in the help
    parser.add_argument('entry_point', type=str, help='The path to the model python file')
    parser.add_argument('--num_process', type=int, help='The number of parallel training processes',
                        default=max(1, get_gpu_count()))
    parser.add_argument('--inputs', type=str, help='The path where tfrecord is saved or will be saved')
    parser.add_argument('--hyperparameters', dest='hyperparameters_json', type=str,
                        help="The path to the hyperparameters JSON file")
    parser.add_argument_group(
        'hyperparameter arguments', 'Arguments to be passed through to the get_estimator() call. \
        Examples might look like --epochs <int>, --batch_size <int>, --optimizer <str>, etc...')
    parser.set_defaults(func=train)
