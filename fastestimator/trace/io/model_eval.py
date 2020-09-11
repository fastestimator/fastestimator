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
import locale
import os
import re
import shutil
from datetime import datetime
from time import time
from typing import Any, Callable, List, Union

import numpy as np
from pylatex import Command, Document, Itemize, MultiColumn, NoEscape, Package, Section, Subsection, Table, Tabularx, \
    escape_latex

import fastestimator as fe
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.latex_util import TabularCell, WrapText
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list, to_number


@traceable()
class TestCase():
    """This class defines the test case that ModelEval trace will take to perform auto-testing.

    Args:
        description: A test description.
        criteria: Function to perform the test returns True when the test passes and False when it fails. Input
            variable names will be used as input keys to further derive input value.
        aggregate: If True, this test is aggregate type and its criteria function will be examined at epoch_end. If
            False, this test is per-instance type and its test criteria function will be examined at batch_end.
        fail_threshold: Thershold of failure instance number to judge per-instance test as failed or passed. If failure
            number is above this value, then the test fails; otherwise it passes. It only has effect when aggregate is
            equal to False.
    """
    def __init__(self, description: str, criteria: Callable, aggregate: bool = True, fail_threshold: int = 0) -> None:
        self.description = description
        self.criteria = criteria
        self.criteria_inputs = inspect.signature(criteria).parameters.keys()
        self.aggregate = aggregate
        if not self.aggregate:
            self.fail_threshold = fail_threshold

    def clean_result(self) -> None:
        if not self.aggregate:
            self.result = []
            self.fail_id = []


@traceable()
class ModelEval(Trace):
    """Automate testing and report generation.

    This trace will examine all input TestCases in the test mode and generate a PDF report and a JSON test result.


    Args:
        test_cases: TestCase object or list of TestCase objects.
        save_path: Where to save the output directory.
        test_title: Title of the test.
        data_id: Data instance ID key. If provided, then per-instance test will return failure instance ID.
    """
    def __init__(self,
                 test_cases: Union[TestCase, List[TestCase]],
                 save_path: str,
                 test_title: str = "Test",
                 data_id: str = None) -> None:

        self.test_title = test_title
        self.test_cases = to_list(test_cases)
        self.instance_cases = []
        self.aggregate_cases = []
        self.data_id = data_id

        all_inputs = set()
        for case in self.test_cases:
            all_inputs.update(case.criteria_inputs)
            case.clean_result()
            if case.aggregate:
                self.aggregate_cases.append(case)
            else:
                self.instance_cases.append(case)

        if self.data_id:
            all_inputs.update([data_id])

        path = os.path.normpath(save_path)
        path = os.path.abspath(path)
        root_dir = os.path.dirname(path)
        report = os.path.basename(path) or 'report'
        report = report.split('.')[0]
        self.save_dir = os.path.join(root_dir, report)
        self.src_dir = os.path.join(self.save_dir, "resources")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.src_dir, exist_ok=True)

        super().__init__(inputs=all_inputs, mode="test")

    def on_begin(self, data: Data) -> None:
        self._sanitize_report_name()
        self._initialize_json_summary()

    def on_batch_end(self, data: Data) -> None:
        for case in self.instance_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, np.ndarray):
                raise TypeError(f"In test with description \'{case.description}\': "
                                "Criteria return of per-instance test needs to be ndarray with dtype bool.")

            elif result.dtype != np.dtype("bool"):
                raise TypeError(f"In test with description \'{case.description}\': "
                                "Criteria return of per-instance test needs to be ndarray with dtype bool.")

            result = result.reshape(-1)
            case.result.append(result)
            if self.data_id:
                data_id = to_number(data[self.data_id]).reshape((-1, ))
                if data_id.size != result.size:
                    raise ValueError(f"In test with description \'{case.description}\': "
                                     "Array size of criteria return doesn't match ID array size. Size of criteria"
                                     "return should be equal to the batch_size such that each entry represents test"
                                     "result of corresponding data instance.")

                case.fail_id.append(data_id[result == False])

    def on_epoch_end(self, data: Data) -> None:
        for case in self.aggregate_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, (bool, np.bool_)):
                raise TypeError(f"In test with description \'{case.description}\': "
                                "Criteria return of aggregate-case test needs to be a bool.")
            case.result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            case.input_val = {var_name: self._to_serializable(data[var_name]) for var_name in case.criteria_inputs}

    def on_end(self, data: Data) -> None:
        for case in self.instance_cases:
            case_dict = {"test_type": "per-instance", "description": case.description}
            result = np.hstack(case.result)
            fail_num = np.sum(result == False)
            case_dict["passed"] = self._to_serializable(fail_num <= case.fail_threshold)
            case_dict["fail_threshold"] = case.fail_threshold
            case_dict["fail_number"] = self._to_serializable(fail_num)
            if self.data_id:
                fail_id = np.hstack(case.fail_id)
                case_dict["fail_id"] = self._to_serializable(fail_id)
            self.json_summary["tests"].append(case_dict)

        for case in self.aggregate_cases:
            case_dict = {"test_type": "aggregate", "description": case.description}
            case_dict["passed"] = self._to_serializable(case.result)
            case_dict["inputs"] = case.input_val
            self.json_summary["tests"].append(case_dict)

        self.json_summary["execution_time(s)"] = time() - self.json_summary["execution_time(s)"]

        self._dump_json()
        self._init_document()
        self._document_test_result()
        self._dump_pdf()

    def _initialize_json_summary(self) -> None:
        """Initialize json summary
        """
        self.json_summary = {
            "title": self.test_title, "timestamp": str(datetime.now()), "execution_time(s)": time(), "tests": []
        }

    def _sanitize_report_name(self) -> None:
        """Sanitize report name and make it class attribute
        """
        exp_name = self.system.summary.name
        if not exp_name:
            raise RuntimeError("TestReport require an experiment name to be provided in estimator.fit()")
        # Convert the experiment name to a report name (useful for saving multiple experiments into same directory)
        report_name = "".join('_' if c == ' ' else c for c in exp_name
                              if c.isalnum() or c in (' ', '_')).rstrip("_").lower()
        report_name = re.sub('_{2,}', '_', report_name) + "_ModelEval"
        self.report_name = report_name or '_ModelEval'

    def _init_document(self) -> None:
        """Initialize latex document
        """
        self.doc = Document(geometry_options=['lmargin=2cm', 'rmargin=2cm', 'tmargin=2cm', 'bmargin=2cm'])
        self.doc.packages.append(Package(name='placeins', options=['section']))
        self.doc.packages.append(Package(name='float'))

        self.doc.preamble.append(NoEscape(r'\aboverulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\belowrulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\renewcommand{\arraystretch}{1.2}'))

        self.doc.preamble.append(Command('title', self.json_summary["title"]))
        self.doc.preamble.append(Command('author', f"FastEstimator {fe.__version__}"))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

        # new column type for tabularx
        self.doc.preamble.append(NoEscape(r'\newcolumntype{Y}{>{\centering\arraybackslash}X}'))

        # add seqinsert hyphentation
        self.doc.preamble.append(NoEscape(r'\def\seqinsert{\-}'))

        # TOC
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.doc.append(NoEscape(r'\newpage'))

    def _document_test_result(self) -> None:
        """Document test results including test summary, passed tests, and failed tests.
        """
        self.test_id = 1
        instance_pass_tests, aggregate_pass_tests, instance_fail_tests, aggregate_fail_tests = [], [], [], []

        for test in self.json_summary["tests"]:
            if test["test_type"] == "per-instance" and test["passed"] == True:
                instance_pass_tests.append(test)
            elif test["test_type"] == "per-instance" and test["passed"] == False:
                instance_fail_tests.append(test)
            elif test["test_type"] == "aggregate" and test["passed"] == True:
                aggregate_pass_tests.append(test)
            elif test["test_type"] == "aggregate" and test["passed"] == False:
                aggregate_fail_tests.append(test)

        with self.doc.create(Section("Test Summary")):
            with self.doc.create(Itemize()) as itemize:
                itemize.add_item(
                    escape_latex("Execution time: {:.2f} seconds".format(self.json_summary['execution_time(s)'])))

            with self.doc.create(Table(position='H')) as table:
                table.append(NoEscape(r'\refstepcounter{table}'))
                self._document_summary_table(pass_num=len(instance_pass_tests) + len(aggregate_pass_tests),
                                             fail_num=len(instance_fail_tests) + len(aggregate_fail_tests))

        if instance_fail_tests or aggregate_fail_tests:
            with self.doc.create(Section("Failed Tests")):
                if len(aggregate_fail_tests) > 0:
                    with self.doc.create(Subsection("Failed Aggregate Tests")):
                        with self.doc.create(Table(position='H')) as table:
                            table.append(NoEscape(r'\refstepcounter{table}'))
                            self._document_aggregate_table(tests=aggregate_fail_tests)
                if len(instance_fail_tests) > 0:
                    with self.doc.create(Subsection("Failed Per-Instance Tests")):
                        with self.doc.create(Table(position='H')) as table:
                            table.append(NoEscape(r'\refstepcounter{table}'))
                            self._document_instance_table(tests=instance_fail_tests, with_ID=self.data_id)

        if instance_pass_tests or aggregate_pass_tests:
            with self.doc.create(Section("Passed Tests")):
                if aggregate_pass_tests:
                    with self.doc.create(Subsection("Passed Aggregate Tests")):
                        with self.doc.create(Table(position='H')) as table:
                            table.append(NoEscape(r'\refstepcounter{table}'))
                            self._document_aggregate_table(tests=aggregate_pass_tests)
                if instance_pass_tests:
                    with self.doc.create(Subsection("Passed Per-Instance Tests")):
                        with self.doc.create(Table(position='H')) as table:
                            table.append(NoEscape(r'\refstepcounter{table}'))
                            self._document_instance_table(tests=instance_pass_tests, with_ID=self.data_id)

    def _document_summary_table(self, pass_num: int, fail_num: int) -> None:
        """Document a summary table.

        Args:
            pass_num: Total number of passed tests.
            fail_num: Total number of failed tests.
        """
        with self.doc.create(Tabularx('|Y|Y|Y|', booktabs=True)) as tabular:
            package = Package('seqsplit')
            if package not in tabular.packages:
                tabular.packages.append(package)

            # add table heading
            tabular.add_row(("Total test #", "Total pass #", "Total fail #"), strict=False)
            tabular.add_hline()

            tabular.add_row((pass_num + fail_num, pass_num, fail_num), strict=False)

    def _document_instance_table(self, tests: List[dict], with_ID: bool):
        """Document a result table of per-instance tests.

        Args:
            tests: List of corresponding test dictionary to make a table.
            with_ID: Whether the test information includes data ID.
        """
        if with_ID:
            table_spec = '|c|X|c|c|X|'
        else:
            table_spec = '|c|X|c|c|'

        with self.doc.create(Tabularx(table_spec, booktabs=True)) as tabular:
            package = Package('seqsplit')
            if package not in tabular.packages:
                tabular.packages.append(package)

            # add table heading
            row_cells = [
                MultiColumn(size=1, align='|c|', data="Test ID"),
                MultiColumn(size=1, align='c|', data="Test description"),
                MultiColumn(size=1, align='c|', data="Pass threshold"),
                MultiColumn(size=1, align='c|', data="Failure count")
            ]

            if with_ID:
                row_cells.append(MultiColumn(size=1, align='c|', data="Failure data instance ID"))

            tabular.add_row(row_cells)
            tabular.add_hline()

            for idx, test in enumerate(tests, 1):
                if idx > 1:
                    tabular.add_hline()

                des_data = [WrapText(data=x, seq_thld=27) for x in test["description"].split(" ")]
                row_cells = [
                    self.test_id,
                    TabularCell(data=des_data, token=" "),
                    NoEscape(r'$\le $' + str(test["fail_threshold"])),
                    test["fail_number"]
                ]
                if with_ID:
                    id_data = [WrapText(data=x, seq_thld=27) for x in test["fail_id"]]
                    row_cells.append(TabularCell(data=id_data, token=", "))

                tabular.add_row(row_cells)
                self.test_id += 1

    def _document_aggregate_table(self, tests: List[dict]) -> None:
        """Document a result table of aggregate tests.

        Args:
            tests: List of corresponding test dictionary to make a table.
        """
        with self.doc.create(Tabularx('|c|X|X|', booktabs=True)) as tabular:
            package = Package('seqsplit')
            if package not in tabular.packages:
                tabular.packages.append(package)

            # add table heading
            tabular.add_row((MultiColumn(size=1, align='|c|', data="Test ID"),
                             MultiColumn(size=1, align='c|', data="Test description"),
                             MultiColumn(size=1, align='c|', data="Input value")))
            tabular.add_hline()

            for idx, test in enumerate(tests, 1):
                if idx > 1:
                    tabular.add_hline()

                inp_data = ["{}={}".format(arg, value) for arg, value in test["inputs"].items()]
                inp_data = [WrapText(data=x, seq_thld=27) for x in inp_data]
                des_data = [WrapText(data=x, seq_thld=27) for x in test["description"].split(" ")]

                row_cells = [
                    self.test_id,
                    TabularCell(data=des_data, token=" "),
                    TabularCell(data=inp_data, token=escape_latex(", \n")),
                ]

                tabular.add_row(row_cells)
                self.test_id += 1

    def _dump_pdf(self) -> None:
        """Dump PDF summary report.
        """
        # self.doc.dumps_packages()
        if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
            # No LaTeX Compiler is available
            self.doc.generate_tex(os.path.join(self.save_dir, self.report_name))
            suffix = '.tex'
        else:
            # Force a double-compile since some compilers will struggle with TOC generation
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False, clean=False)
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False)
            suffix = '.pdf'
        print("FastEstimator-TestReport: Report written to {}{}".format(os.path.join(self.save_dir, self.report_name),
                                                                        suffix))

    def _dump_json(self) -> None:
        """Dump JSON file
        """
        json_path = os.path.join(self.src_dir, self.report_name + ".json")
        with open(json_path, 'w') as fp:
            json.dump(self.json_summary, fp, indent=4)

    @staticmethod
    def _to_serializable(obj: Any) -> Union[float, int, list]:
        """Convert to JSON serializable type.

        Args:
            obj: Any object that needs to be converted.

        Return:
            JSON serializable object that essentially is equivalent to input obj.
        """
        if isinstance(obj, np.ndarray):
            shape = obj.shape
            obj = obj.reshape((-1, ))
            obj = np.vectorize(ModelEval._element_to_serializable)(obj)
            obj = obj.reshape(shape)
            obj = obj.tolist()

        else:
            obj = ModelEval._element_to_serializable(obj)

        return obj

    @staticmethod
    def _element_to_serializable(obj):
        """Convert to JSON serializable type.

        This function can handle any object type except ndarray.

        Args:
            obj: Any object except ndarray that needs to be converted.

        Return:
            JSON serializable object that essentially is equivalent to input obj.
        """
        if isinstance(obj, bytes):
            obj = obj.decode('utf-8')

        elif isinstance(obj, np.generic):
            obj = obj.item()

        return obj

    @staticmethod
    def check_pdf_dependency() -> None:
        """Check dependency of PDF-generating packages.

        Raises:
            OSError: Some required package has not been installed.
        """
        # Verify that the system locale is functioning correctly
        try:
            locale.getlocale()
        except ValueError:
            raise OSError("Your system locale is not configured correctly. On mac this can be resolved by adding \
                'export LC_ALL=en_US.UTF-8' and 'export LANG=en_US.UTF-8' to your ~/.bash_profile")
