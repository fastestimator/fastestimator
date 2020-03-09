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
import pdb
from typing import Callable, List, Union

from docx import Document

from fastestimator.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.util import to_list


class QMSTest(Trace):
    def __init__(self,
                 test_descriptions: Union[str, List[str]],
                 test_criterias: Union[List[Callable], Callable],
                 test_title: str = "QMSTest",
                 output_path: str = "",):
        self.output_path = output_path
        self.test_title = test_title
        self.test_descriptions = to_list(test_descriptions)
        self.test_criterias = to_list(test_criterias)
        assert len(self.test_descriptions) == len(self.test_criterias), "inconsistent input length found"
        all_inputs = set()
        for criteria in self.test_criterias:
            all_inputs.update(inspect.signature(criteria).parameters.keys())
        super().__init__(inputs=all_inputs, mode="test")

    def _initialize_json_summary(self):
        self.json_summary = {"title": self.test_title, "stories": []}

    def _initialize_doc_summary(self):
        self.doc_summary = Document()
        doc_table = self.doc_summary.add_table(rows=2, cols=4)
        doc_table.rows[0].cells[0].text = "Model"
        doc_table.rows[0].cells[1].text = "Total Tests"
        doc_table.rows[0].cells[2].text = "Tests Passed"
        doc_table.rows[0].cells[3].text = "Tests Failed"
        doc_table.rows[1].cells[0].text = "Model#1"
        doc_table.rows[1].cells[1].text = str(len(self.test_criterias))

        self.total_pass, self.total_fail = 0, 0
        self.doc_table = doc_table

    def on_begin(self, data: Data):
        self._initialize_json_summary()
        self._initialize_doc_summary()

    def on_epoch_end(self, data: Data):
        for criteria, description in zip(self.test_criterias, self.test_descriptions):
            story = {"description": description}
            is_passed = criteria(*[data[var_name] for var_name in list(inspect.signature(criteria).parameters.keys())])
            story["passed"] = str(is_passed)

            if is_passed:
                self.total_pass += 1
            else:
                self.total_fail += 1

            self.json_summary["stories"].append(story)

        self.doc_table.rows[1].cells[2].text = str(self.total_pass)
        self.doc_table.rows[1].cells[3].text = str(self.total_fail)

    def on_end(self, data: Data):
        if self.output_path.endswith(".json"):
            json_path = self.output_path
        else:
            json_path = os.path.join(self.output_path, "{}.json".format("QMS"))
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)

        self.doc_summary.save("summary_report.docx")
        print("Saved QMS report to {}".format(json_path))
