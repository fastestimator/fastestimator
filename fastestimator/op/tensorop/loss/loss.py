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
from fastestimator.op import TensorOp


class Loss(TensorOp):
    """A base class for loss operations. It can be used directly to perform value pass-through (see the adversarial
    training showcase for an example of when this is useful)
    """
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    @staticmethod
    def validate_loss_inputs(inputs, *args):
        """A method to ensure that either the inputs array or individual input arguments are specified, but not both
        Args:
            inputs: None or a tuple/list of arguments
            *args: a tuple of arguments or Nones
        Returns:
            either 'inputs' or the args tuple depending on which is populated
        """
        if inputs is None:  # Using args
            assert all(map(lambda x: x is not None, args)), \
                "If the 'inputs' field is not provided then all individual input arguments must be specified"
            inputs = args
        else:  # Using Inputs
            assert all(map(lambda x: x is None, args)), \
                "If the 'inputs' field is provided then individual input arguments may not be specified"
            assert len(inputs) == len(args), \
                "{} inputs were provided, but {} were required".format(len(inputs), len(args))
        return inputs
