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
import inspect
import json
import os
from typing import Callable, List, Union

from fastestimator.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.util import to_list


class QMSTest(Trace):
    def __init__(self,
                 test_descriptions: Union[str, List[str]],
                 test_criterias: Union[List[Callable], Callable],
                 test_title: str = "QMSTest",
                 output_path: str = ""):
        self.output_path = output_path
        self.test_title = test_title
        self.test_descriptions = to_list(test_descriptions)
        self.test_criterias = to_list(test_criterias)
        self._initialize_summary()
        assert len(self.test_descriptions) == len(self.test_criterias), "inconsistent input length found"
        all_inputs = set()
        for criteria in self.test_criterias:
            all_inputs.update(inspect.signature(criteria).parameters.keys())
        super().__init__(inputs=all_inputs, mode="test")

    def _initialize_summary(self):
        self.json_summary = {"title": self.test_title, "stories": []}

    def on_begin(self, data: Data):
        self._initialize_summary()

    def on_epoch_end(self, data: Data):
        for criteria, description in zip(self.test_criterias, self.test_descriptions):
            story = {"description": description}
            story["passed"] = str(
                criteria(*[data[var_name] for var_name in list(inspect.signature(criteria).parameters.keys())]))
            self.json_summary["stories"].append(story)

    def on_end(self, data: Data):
        if self.output_path.endswith(".json"):
            json_path = self.output_path
        else:
            json_path = os.path.join(self.output_path, "{}.json".format("QMS"))
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)
        print("Saved QMS report to {}".format(json_path))
