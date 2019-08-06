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
import os

from pyfiglet import Figlet


class SaveAction(argparse.Action):
    """
    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
        is invoked directly during argument parsing.
    """

    def __init__(self, option_strings, dest, nargs='?', **kwargs):
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super(SaveAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))


def draw():
    print(Figlet(font="slant").renderText("FastEstimator"))
