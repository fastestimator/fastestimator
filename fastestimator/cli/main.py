#!/usr/bin/env python
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

from fastestimator.cli.train import configure_train_parser, configure_test_parser
from fastestimator.cli.logs import configure_log_parser


def run() -> None:
    """A function which invokes the various argument parsers and then runs the requested subroutine.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers()
    # In python 3.7 the following 2 lines could be put into the .add_subparsers() call
    subparsers.required = True
    subparsers.dest = 'mode'
    configure_train_parser(subparsers)
    configure_test_parser(subparsers)
    configure_log_parser(subparsers)
    args, unknown = parser.parse_known_args()
    args.func(vars(args), unknown)


if __name__ == '__main__':
    run()
