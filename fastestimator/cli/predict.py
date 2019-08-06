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


def predict(args, unknown):
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    print("Prediction not yet supported, but coming soon...!")


def configure_predict_parser(subparsers):
    parser = subparsers.add_parser('predict', description='Evaluate a trained model on new inputs', allow_abbrev=False)
    parser.set_defaults(func=predict)
