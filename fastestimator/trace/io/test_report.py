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
from datetime import datetime
from time import time
from typing import Callable, List, Union

import numpy as np

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import get_environment, traceable
from fastestimator.util.util import to_list, to_number


@traceable()
class TestCase():
    """This class defines the test case that TestReport trace will take to perform auto-testing.

    Args:
        description: A test description.
        criteria: Function to perform the test that return True when test passes and False when test fails. Input
            variable name will be used as input keys to futher derive input value.
        sample_wise: If True, this test will be treated as sample-case of which test criteria will be examined at
            batch_end. If False, this test is epoch-case and its criteria will be examined at epoch_end.
        fail_threshold: Thershold of failing sample number to judge sample-case test as failed or passed. If failing
            number is above this value, then the test fails; otherwise it passes. It only has effect when sample_wise
            equal to true.
    """
    def __init__(self, description: str, criteria: Callable, sample_wise: bool = False,
                 fail_threshold: int = 0) -> None:
        self.description = description
        self.criteria = criteria
        self.criteria_inputs = inspect.signature(criteria).parameters.keys()
        self.sample_wise = sample_wise
        if self.sample_wise:
            self.fail_threshold = fail_threshold

    def clean_result(self) -> None:
        if self.sample_wise:
            self.result = []
            self.fail_id = []


@traceable()
class TestReport(Trace):
    """Automate testing and report generation.

    Args:
        test_cases: List of TestCase object.
        test_title: Title of the test.
        json_output: Path into which to write the output results JSON.
        sample_id: Data sample ID key. If provided, then sample-case test will return failing sample ID.
    """
    def __init__(self,
                 test_cases: Union[TestCase, List[TestCase]],
                 test_title: str = "Test",
                 json_output: str = "",
                 sample_id=None) -> None:

        self.json_output = json_output
        self.test_title = test_title
        self.test_cases = to_list(test_cases)
        self.sample_cases = []
        self.epoch_cases = []
        self.sample_id = sample_id
        all_inputs = set()
        for case in self.test_cases:
            all_inputs.update(case.criteria_inputs)
            case.clean_result()
            if case.sample_wise:
                self.sample_cases.append(case)
            else:
                self.epoch_cases.append(case)

        if self.sample_id:
            all_inputs.update([sample_id])

        super().__init__(inputs=all_inputs, mode="test")

    def _initialize_json_summary(self) -> None:
        """Initialize json summary
        """
        self.json_summary = {
            "title": self.test_title, "timestamp": str(datetime.now()), "execution_time(s)": time(), "tests": []
        }

    def on_begin(self, data: Data) -> None:
        self._initialize_json_summary()

    def on_batch_end(self, data: Data) -> None:
        for case in self.sample_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, np.ndarray):
                raise TypeError("Criteria return of sample-case test need to be ndarray with dtype bool")
            elif result.dtype != np.dtype("bool"):
                raise TypeError("Criteria return of sample-case test need to be ndarray with dtype bool")

            result = result.reshape(-1)
            case.result.append(result)
            if self.sample_id:
                data_id = to_number(data[self.sample_id]).reshape((-1, ))
                if data_id.size != result.size:
                    raise ValueError("Array size of criteria return doesn't match ID array size."
                                     "Criteria return size should be equal to the batch_size that each entry represents"
                                     "test result of corresponding sample")
                case.fail_id.append(data_id[result == False])

    def on_epoch_end(self, data: Data) -> None:
        for case in self.epoch_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, (bool, np.bool_)):
                raise TypeError("criteria return of epoch-case test need to be bool")
            case.result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            case.input_val = {var_name: self._to_serializable(data[var_name]) for var_name in case.criteria_inputs}

    def on_end(self, data: Data) -> None:
        for case in self.sample_cases:
            case_dict = {"test_type": "sample", "description": case.description}
            result = np.hstack(case.result)
            fail_num = np.sum(result == False)
            case_dict["passed"] = self._to_serializable(fail_num <= case.fail_threshold)
            case_dict["fail_threshold"] = case.fail_threshold
            case_dict["fail_number"] = self._to_serializable(fail_num)
            if self.sample_id:
                fail_id = np.hstack(case.fail_id)
                case_dict["fail_id"] = self._to_serializable(fail_id)
            self.json_summary["tests"].append(case_dict)

        for case in self.epoch_cases:
            case_dict = {"test_type": "epoch", "description": case.description}
            case_dict["passed"] = self._to_serializable(case.result)
            case_dict["inputs"] = case.input_val
            self.json_summary["tests"].append(case_dict)

        self.json_summary["execution_time(s)"] = time() - self.json_summary["execution_time(s)"]
        self.json_summary["environment"] = get_environment()

        if self.json_output.endswith(".json"):
            json_path = self.json_output
        else:
            json_path = os.path.join(self.json_output, "test_report.json")
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)
        print("Saved test JSON report to {}".format(json_path))

    @staticmethod
    def _to_serializable(obj: np.generic) -> Union[float, int, list]:
        """ convert to JSON serializable type
        """
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()

        elif isinstance(obj, np.generic):
            obj = np.asscalar(obj)

        return obj
