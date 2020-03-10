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

from docx import Document
from docx.shared import Pt

from fastestimator.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.util import to_list


class QMSTest(Trace):
    def __init__(self,
                 test_descriptions: Union[str, List[str]],
                 test_criterias: Union[List[Callable], Callable],
                 test_title: str = "QMSTest",
                 json_output: str = "",
                 doc_output: str = ""):

        self.json_output = json_output
        self.doc_output = doc_output
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

    def on_begin(self, data: Data):
        self._initialize_json_summary()
        self.total_pass, self.total_fail = 0, 0

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

    def on_end(self, data: Data):
        if self.json_output.endswith(".json"):
            json_path = self.json_output
        else:
            json_path = os.path.join(self.output_path, "QMS.json")
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)
        print("Saved QMS JSON report to {}".format(json_path))

        if self.doc_output.endswith(".docx"):
            doc_path = self.doc_output
        else:
            doc_path = os.path.join(self.doc_output, "QMS_summary.docx")

        doc_summary = QMSDocx(self.total_pass, self.total_fail)
        doc_summary.save(doc_path)
        print("Saved QMS summary report to {}".format(doc_path))


class QMSDocx():
    def __init__(self, total_pass, total_fail):
        self.doc = Document()
        self._write_static_p1()
        self._write_test_result(total_pass, total_fail)
        self._write_static_p2()

    def save(self, output_path):
        self.doc.save(output_path)

    def _write_test_result(self, total_pass, total_fail):
        total_test = total_pass + total_fail

        T = self.doc.add_table(rows=2, cols=4)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()
                if i == 0:
                    A.bold = True
        T.style = "Table Grid"
        self.fill_table(T,
                        [["Model", "Total Tests", "Tests Passed", "Tests Failed"],
                         ["Model#1", str(total_test), str(total_pass), str(total_fail)]])

    def _write_static_p1(self):
        P = self.doc.add_paragraph()
        self.add_line_break(P, 9, font_size=Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(22)
        A.add_text("Verification Summary report for Model")
        self.add_line_break(P, 4, font_size=Pt(14))

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("Record of changes")
        A.add_break()

        T = self.doc.add_table(rows=2, cols=4)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()
                A.bold = True
                A.font.size = Pt(14)

        T.style = "Table Grid"
        self.fill_table(T, [["Rev number", "Date", "Author", "Comments"], ["1", "", "<Name>", "Initial revision"]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 3, Pt(14))
        A = P.add_run()
        A.bold = True
        A.add_text("NOTE:")
        A = P.add_run()
        A.add_text(" Copies for use are available via the MyWorkshop system. Any printed copies are considered" \
        " uncontrolled. All approval signatures are captured electronically in MyWorkshop.")
        self.add_line_break(P, 9, Pt(14))

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("1   Introdction")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.bold = True
        A.add_text("1.1 Purpose & Scope")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text(
            "This document contains the results of the verification for model for <name>. Tests were executed in " \
            "accordance with the associated Verification Plan (DOC). ")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.bold = True
        A.add_text("1.2 References")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text("Find below all the relevant documents related to any tools used during verification. ")

        T = self.doc.add_table(rows=4, cols=3)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()
                if i == 0:
                    A.bold = True

        T.style = "Table Grid"
        self.fill_table(
            T,
            [["Location", "Reference", "Document Name"], ["Myworkshop", "<DOC>", "Edison AI Model Verification Plan"], [
                "Myworkshop", "<DOC>", "Model Evaluation Tool Validation"
            ], ["Myworkshop", "<DOC>", "The CRS documents are in Approved state"]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("2   Infrastructure Details")

        T = self.doc.add_table(rows=6, cols=2)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()
                if i == 0:
                    A.bold = True
        T.style = "Table Grid"
        self.fill_table(T,
                        [["", "Details"], ["GPU Architecture", ""], ["OS environment", ""], [
                            "Collection ID of test data set in Edison AI Workbench", ""
                        ], ["Model Artifact ID (s)", ""], ["Location of model test scripts", ""]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("3   Verification Tools")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text("Tools used for verification are listed in Model Evaluation Tool Validation <DOC>")

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("4   Results of Verification")

        T = self.doc.add_table(rows=2, cols=3)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()

        T.style = "Table Grid"
        self.fill_table(T,
                        [["Document", "Location", "Comments"],
                         ["<DOC>", "Myworkshop", "Verification Procedure is in Approved state"]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 1, Pt(14))
        A = P.add_run()
        A.bold = True
        A.add_text("4.1 Functional and Performance Tests")

    def _write_static_p2(self):
        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("5   Verification Details")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text("Below table details the summary of the completion of verification cycle. ")

        T = self.doc.add_table(rows=9, cols=2)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()

                if i == 0 or j == 0:
                    A.bold = True

        T.style = "Table Grid"
        self.fill_table(
            T,
            [["Activity", "Details"],
             [
                 "Test set location in ALM",
                 "URL: http://hc-alm12.health.ge.com/qcbin/start_a.jsp " \
                 "Domain: SWPE / Project: HealthCloud \n ALM\Test Lab\<location of ALM test set>"],
             ["Verification Cycle Start Date", ""], ["Verification Cycle End Date", ""],
             ["Name of the Tester(s)", ""], ["Total # of test cases executed", ""], ["Total # of Defects Filed", ""],
             ["Total # of Tests Passed", ""], ["Total # of Tests Failed", ""]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("6   Defect Summary List")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text("Below table summarizes the defects found during verification cycle. " \
                   "The defects are tracked in ALM: http://hc-alm12.health.ge.com/qcbin/start_a.jsp " \
                   "Domain: SWPE / Project: HealthCloud")

        T = self.doc.add_table(rows=2, cols=5)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()
                if i == 0:
                    A.bold = True

        T.style = "Table Grid"
        self.fill_table(T,
                        [["Defect ID", "Summary", "Classification", "Status", "Justification"], ["", "", "", "", ""]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("7   Verification Deviations")

        T = self.doc.add_table(rows=2, cols=1)
        for i in range(len(T.rows)):
            for j in range(len(T.columns)):
                A = T.rows[i].cells[j].paragraphs[0].add_run()

        T.style = "Table Grid"
        self.fill_table(T,
                        [["There were no deviations from the verification plan."],
                         ["There were deviations from the verification plan as follows"]])

        P = self.doc.add_paragraph()
        self.add_line_break(P, 2, Pt(14))
        A = P.add_run()
        A.bold = True
        A.font.size = Pt(14)
        A.add_text("8   Conclusion")

        P = self.doc.add_paragraph()
        A = P.add_run()
        A.add_text("The acceptance criteria identified in the Verification plan have been met. All activities " \
                   "supporting this verification activity are complete.")

    @staticmethod
    def fill_table(table, content):
        assert len(table.rows) == len(content)
        assert len(table.columns) == len(content[0])

        for i in range(len(table.rows)):
            for j in range(len(table.columns)):
                table.rows[i].cells[j].paragraphs[0].runs[0].add_text(content[i][j])

    @staticmethod
    def add_line_break(paragraph, num, font_size=None):
        run = paragraph.add_run()
        if font_size:
            run.font.size = font_size

        for i in range(num):
            run.add_break()
