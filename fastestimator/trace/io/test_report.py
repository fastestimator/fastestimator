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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pylatex import Command, Document, Itemize, LongTable, MultiColumn, NoEscape, Package, Section, Subsection, Table, \
    Tabularx, escape_latex

import fastestimator as fe
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.latex_util import IterJoin, WrapText
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list, to_number, to_set


@traceable()
class TestCase:
    """This class defines the test case that the TestReport trace will take to perform auto-testing.

    Args:
        description: A test description.
        criteria: A function to perform the test. For an aggregate test, `criteria` needs to return True when the test
            passes and False when it fails. For a per-instance test, `criteria` needs to return a boolean np.ndarray,
            where entries show corresponding test results (True if the test of that data instance passes; False if it
            fails).
        aggregate: If True, this test is aggregate type and its `criteria` function will be examined at epoch_end. If
            False, this test is per-instance type and its `criteria` function will be examined at batch_end.
        fail_threshold: Threshold of failure instance number to judge the per-instance test as failed or passed. If
            the failure number is above this value, then the test fails; otherwise it passes. It can only be set when
            `aggregate` is equal to False.

    Raises:
        ValueError: If user set `fail_threshold` for an aggregate test.
    """
    def __init__(self,
                 description: str,
                 criteria: Callable[..., Union[bool, np.ndarray]],
                 aggregate: bool = True,
                 fail_threshold: int = 0) -> None:
        self.description = description
        self.criteria = criteria
        self.criteria_inputs = inspect.signature(criteria).parameters.keys()
        self.aggregate = aggregate
        if self.aggregate:
            if fail_threshold:
                raise ValueError("fail_threshold cannot be set in a aggregate test")
        else:
            self.fail_threshold = fail_threshold
        self.result = None
        self.input_val = None
        self.fail_id = []
        self.init_result()

    def init_result(self) -> None:
        """Reset the test result.
        """
        if self.aggregate:
            self.result = None
            self.input_val = None
        else:
            self.result = []
            self.fail_id = []


@traceable()
class TestReport(Trace):
    """Automate testing and report generation.

    This trace will evaluate all its `test_cases` during test mode and generate a PDF report and a JSON test result.

    Args:
        test_cases: The test(s) to be run.
        save_path: Where to save the outputs.
        test_title: The title of the test, or None to use the experiment name.
        data_id: Data instance ID key. If provided, then per-instances test will include failing instance IDs.
    """
    def __init__(self,
                 test_cases: Union[TestCase, List[TestCase]],
                 save_path: str,
                 test_title: Optional[str] = None,
                 data_id: str = None) -> None:

        self.check_pdf_dependency()

        self.test_title = test_title
        self.report_name = None

        self.instance_cases = []
        self.aggregate_cases = []
        self.data_id = data_id

        all_inputs = to_set(self.data_id)
        for case in to_list(test_cases):
            all_inputs.update(case.criteria_inputs)
            if case.aggregate:
                self.aggregate_cases.append(case)
            else:
                self.instance_cases.append(case)

        path = os.path.normpath(save_path)
        path = os.path.abspath(path)
        root_dir = os.path.dirname(path)
        report = os.path.basename(path) or 'report'
        report = report.split('.')[0]
        self.save_dir = os.path.join(root_dir, report)
        self.resource_dir = os.path.join(self.save_dir, "resources")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.resource_dir, exist_ok=True)

        self.json_summary = {}
        # PDF document related
        self.doc = None
        self.test_id = None

        super().__init__(inputs=all_inputs, mode="test")

    def on_begin(self, data: Data) -> None:
        self._sanitize_report_name()
        self._initialize_json_summary()
        for case in self.instance_cases + self.aggregate_cases:
            case.init_result()

    def on_batch_end(self, data: Data) -> None:
        for case in self.instance_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, np.ndarray):
                raise TypeError(f"In test with description '{case.description}': "
                                "Criteria return of per-instance test needs to be ndarray with dtype bool.")
            elif result.dtype != np.dtype("bool"):
                raise TypeError(f"In test with description '{case.description}': "
                                "Criteria return of per-instance test needs to be ndarray with dtype bool.")
            result = result.reshape(-1)
            case.result.append(result)
            if self.data_id:
                data_id = to_number(data[self.data_id]).reshape((-1, ))
                if data_id.size != result.size:
                    raise ValueError(f"In test with description '{case.description}': "
                                     "Array size of criteria return doesn't match ID array size. Size of criteria"
                                     "return should be equal to the batch_size such that each entry represents the test"
                                     "result of its corresponding data instance.")
                case.fail_id.append(data_id[result == False])

    def on_epoch_end(self, data: Data) -> None:
        for case in self.aggregate_cases:
            result = case.criteria(*[data[var_name] for var_name in case.criteria_inputs])
            if not isinstance(result, (bool, np.bool_)):
                raise TypeError(f"In test with description '{case.description}': "
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
            case_dict = {
                "test_type": "aggregate",
                "description": case.description,
                "passed": self._to_serializable(case.result),
                "inputs": case.input_val
            }
            self.json_summary["tests"].append(case_dict)

        self.json_summary["execution_time(s)"] = time() - self.json_summary["execution_time(s)"]

        self._dump_json()
        self._init_document()
        self._write_body_content()
        self._dump_pdf()

    def _initialize_json_summary(self) -> None:
        """Initialize json summary.
        """
        self.json_summary = {
            "title": self.test_title, "timestamp": str(datetime.now()), "execution_time(s)": time(), "tests": []
        }

    def _sanitize_report_name(self) -> None:
        """Sanitize report name and make it class attribute.

        Raises:
            RuntimeError: If a test title was not provided and the user did not set an experiment name.
        """
        exp_name = self.system.summary.name or self.test_title
        if not exp_name:
            raise RuntimeError("TestReport requires an experiment name to be provided in estimator.fit(), or a title")
        # Convert the experiment name to a report name (useful for saving multiple experiments into same directory)
        report_name = "".join('_' if c == ' ' else c for c in exp_name
                              if c.isalnum() or c in (' ', '_')).rstrip("_").lower()
        self.report_name = re.sub('_{2,}', '_', report_name) + "_TestReport"
        if self.test_title is None:
            self.test_title = exp_name

    def _init_document(self) -> None:
        """Initialize latex document.
        """
        self.doc = self._init_document_geometry()
        self.doc.packages.append(Package(name='placeins', options=['section']))
        self.doc.packages.append(Package(name='float'))
        self.doc.packages.append(Package(name='hyperref', options='hidelinks'))

        self.doc.preamble.append(NoEscape(r'\aboverulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\belowrulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\renewcommand{\arraystretch}{1.2}'))

        # new column type for tabularx
        self.doc.preamble.append(NoEscape(r'\newcolumntype{Y}{>{\centering\arraybackslash}X}'))

        self._write_title()
        self._write_toc()

    def _write_title(self) -> None:
        """Write the title content of the file. Override if you want to build on top of base traceability report.
        """
        self.doc.preamble.append(Command('title', self.json_summary["title"]))
        self.doc.preamble.append(Command('author', f"FastEstimator {fe.__version__}"))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

    def _write_toc(self) -> None:
        """Write the table of contents. Override if you want to build on top of base traceability report.
        """
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.doc.append(NoEscape(r'\newpage'))

    def _write_body_content(self) -> None:
        """Write the main content of the file. Override if you want to build on top of base traceability report.
        """
        self._document_test_result()

    def _document_test_result(self) -> None:
        """Document test results including test summary, passed tests, and failed tests.
        """
        self.test_id = 1
        instance_pass_tests, aggregate_pass_tests, instance_fail_tests, aggregate_fail_tests = [], [], [], []

        for test in self.json_summary["tests"]:
            if test["test_type"] == "per-instance" and test["passed"]:
                instance_pass_tests.append(test)
            elif test["test_type"] == "per-instance" and not test["passed"]:
                instance_fail_tests.append(test)
            elif test["test_type"] == "aggregate" and test["passed"]:
                aggregate_pass_tests.append(test)
            elif test["test_type"] == "aggregate" and not test["passed"]:
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
                        self._document_aggregate_table(tests=aggregate_fail_tests)
                if len(instance_fail_tests) > 0:
                    with self.doc.create(Subsection("Failed Per-Instance Tests")):
                        self._document_instance_table(tests=instance_fail_tests, with_id=bool(self.data_id))

        if instance_pass_tests or aggregate_pass_tests:
            with self.doc.create(Section("Passed Tests")):
                if aggregate_pass_tests:
                    with self.doc.create(Subsection("Passed Aggregate Tests")):
                        self._document_aggregate_table(tests=aggregate_pass_tests)
                if instance_pass_tests:
                    with self.doc.create(Subsection("Passed Per-Instance Tests")):
                        self._document_instance_table(tests=instance_pass_tests, with_id=bool(self.data_id))

        self.doc.append(NoEscape(r'\newpage'))  # For QMS report

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
            tabular.add_row(("Total Tests", "Total Passed ", "Total Failed"), strict=False)
            tabular.add_hline()

            tabular.add_row((pass_num + fail_num, pass_num, fail_num), strict=False)

    def _document_instance_table(self, tests: List[Dict[str, Any]], with_id: bool):
        """Document a result table of per-instance tests.

        Args:
            tests: List of corresponding test dictionary to make a table.
            with_id: Whether the test information includes data ID.
        """
        if with_id:
            table_spec = '|c|p{5cm}|c|c|p{5cm}|'
            column_num = 5
        else:
            table_spec = '|c|p{10cm}|c|c|'
            column_num = 4

        with self.doc.create(LongTable(table_spec, pos=['h!'], booktabs=True)) as tabular:
            package = Package('seqsplit')
            if package not in tabular.packages:
                tabular.packages.append(package)

            # add table heading
            row_cells = [
                MultiColumn(size=1, align='|c|', data="Test ID"),
                MultiColumn(size=1, align='c|', data="Test Description"),
                MultiColumn(size=1, align='c|', data="Pass Threshold"),
                MultiColumn(size=1, align='c|', data="Failure Count")
            ]

            if with_id:
                row_cells.append(MultiColumn(size=1, align='c|', data="Failure Data Instance ID"))

            tabular.add_row(row_cells)

            # add table header and footer
            tabular.add_hline()
            tabular.end_table_header()
            tabular.add_hline()
            tabular.add_row((MultiColumn(column_num, align='r', data='Continued on Next Page'), ))
            tabular.add_hline()
            tabular.end_table_footer()
            tabular.end_table_last_footer()

            for idx, test in enumerate(tests):
                if idx > 0:
                    tabular.add_hline()

                des_data = [WrapText(data=x, threshold=27) for x in test["description"].split(" ")]
                row_cells = [
                    self.test_id,
                    IterJoin(data=des_data, token=" "),
                    NoEscape(r'$\le $' + str(test["fail_threshold"])),
                    test["fail_number"]
                ]
                if with_id:
                    id_data = [WrapText(data=x, threshold=27) for x in test["fail_id"]]
                    row_cells.append(IterJoin(data=id_data, token=", "))

                tabular.add_row(row_cells)
                self.test_id += 1

    def _document_aggregate_table(self, tests: List[Dict[str, Any]]) -> None:
        """Document a result table of aggregate tests.

        Args:
            tests: List of corresponding test dictionary to make a table.
        """
        with self.doc.create(LongTable('|c|p{8cm}|p{7.3cm}|', booktabs=True)) as tabular:
            package = Package('seqsplit')
            if package not in tabular.packages:
                tabular.packages.append(package)

            # add table heading
            tabular.add_row((MultiColumn(size=1, align='|c|', data="Test ID"),
                             MultiColumn(size=1, align='c|', data="Test Description"),
                             MultiColumn(size=1, align='c|', data="Input Value")))

            # add table header and footer
            tabular.add_hline()
            tabular.end_table_header()
            tabular.add_hline()
            tabular.add_row((MultiColumn(3, align='r', data='Continued on Next Page'), ))
            tabular.add_hline()
            tabular.end_table_footer()
            tabular.end_table_last_footer()

            for idx, test in enumerate(tests):
                if idx > 0:
                    tabular.add_hline()

                inp_data = [f"{arg}={self.sanitize_value(value)}" for arg, value in test["inputs"].items()]
                inp_data = [WrapText(data=x, threshold=27) for x in inp_data]
                des_data = [WrapText(data=x, threshold=27) for x in test["description"].split(" ")]
                row_cells = [
                    self.test_id,
                    IterJoin(data=des_data, token=" "),
                    IterJoin(data=inp_data, token=escape_latex(", \n")),
                ]
                tabular.add_row(row_cells)
                self.test_id += 1

    def _dump_pdf(self) -> None:
        """Dump PDF summary report.
        """
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
        """Dump JSON file.
        """
        json_path = os.path.join(self.resource_dir, self.report_name + ".json")
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
            if obj.size > 0:
                shape = obj.shape
                obj = obj.reshape((-1, ))
                obj = np.vectorize(TestReport._element_to_serializable)(obj)
                obj = obj.reshape(shape)

            obj = obj.tolist()

        else:
            obj = TestReport._element_to_serializable(obj)

        return obj

    @staticmethod
    def _element_to_serializable(obj: Any) -> Any:
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

    @staticmethod
    def sanitize_value(value: Union[int, float]) -> str:
        """Sanitize input value for a better report display.

        Args:
            value: Value to be sanitized.

        Returns:
            Sanitized string of `value`.
        """
        if 1000 > value >= 0.001:
            return f"{value:.3f}"
        else:
            return f"{value:.3e}"

    @staticmethod
    def _init_document_geometry() -> Document:
        """Init geometry setting of the document.

        Return:
            Initialized Document object.
        """
        return Document(geometry_options=['lmargin=2cm', 'rmargin=2cm', 'bmargin=2cm'])
