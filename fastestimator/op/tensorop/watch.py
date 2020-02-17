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
from fastestimator.util.util import to_list


class Watch(TensorOp):
    def __init__(self, inputs=None):
        super().__init__(inputs=inputs, outputs=inputs, mode="train")

    def forward(self, data, state):
        """ This class is used to watch variables on the gradient tape

        Args:
            data: input data to watch
            state: Information about the current execution context.
        Returns:
            the input data
        """
        tape = state['tape']
        for elem in to_list(data):
            tape.watch(elem)
        return data
