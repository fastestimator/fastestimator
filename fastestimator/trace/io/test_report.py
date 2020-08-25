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

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list
from time import time
from datetime import datetime
import pdb

@traceable()
class TestReport(Trace):
    """Automate testing and report generation.

    Args:
        test_descriptions: List of text-based descriptions.
        test_criterias: List of test functions. Function input argument names needs to match keys from the data
            dictionary.
        test_title: Title of the test.
        json_output: Path into which to write the output results JSON.
        doc_output: Path into which to write the output QMS summary report (docx).

    Raises:
        AssertionError: If the number of `test_descriptions` and `test_criteria` do not match.
    """
    def __init__(self,
                 test_descriptions: Union[str, List[str]],
                 test_criterias: Union[List[Callable], Callable],
                 test_title: str = "Test",
                 json_output: str = "") -> None:

        self.json_output = json_output
        self.test_title = test_title
        self.test_descriptions = to_list(test_descriptions)
        self.test_criterias = to_list(test_criterias)
        assert len(self.test_descriptions) == len(self.test_criterias), "inconsistent input length found"
        all_inputs = set()
        for criteria in self.test_criterias:
            all_inputs.update(inspect.signature(criteria).parameters.keys())
        super().__init__(inputs=all_inputs, mode="test")

    def _initialize_json_summary(self) -> None:
        """Initialize json summary
        """
        self.json_summary = {
            "title": self.test_title, "timestamp": str(datetime.now()), "execution_time(s)": time(),
            "tests": []
        }

    def on_begin(self, data: Data) -> None:
        self._initialize_json_summary()

    def on_epoch_end(self, data: Data) -> None:
        for criteria, description in zip(self.test_criterias, self.test_descriptions):
            test_case = {"description": description}
            inputs = {var_name: data[var_name] for var_name in list(inspect.signature(criteria).parameters.keys())}
            test_case["inputs"] = inputs

            is_passed = criteria(*[val for val in inputs.values()])
            test_case["passed"] = str(is_passed)
            self.json_summary["tests"].append(test_case)

        self.json_summary["execution_time(s)"] = time() - self.json_summary["execution_time(s)"]

    def on_end(self, data: Data) -> None:
        if self.json_output.endswith(".json"):
            json_path = self.json_output
        else:
            json_path = os.path.join(self.json_output, "test_report.json")
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)
        print("Saved test JSON report to {}".format(json_path))