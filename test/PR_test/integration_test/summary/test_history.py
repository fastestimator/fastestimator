#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from contextlib import closing

from fastestimator.summary.history import HistoryReader, HistoryRecorder, connect, update_settings
from fastestimator.test.unittest_util import sample_system_object


class TestHistoryRecorder(unittest.TestCase):
    def setUp(self):
        self.db_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.db_dir, 'tmp.db')

    def tearDown(self):
        shutil.rmtree(self.db_dir)

    def test_happy_path(self):
        system = sample_system_object()
        recorder = HistoryRecorder(system=system, est_path="test.py", db_path=self.db_path)
        with recorder:
            print("Test Log Capture")
            print("Line 2")
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM history WHERE pk = (?)", [system.exp_id])
            results = cursor.fetchall()
        with self.subTest("History Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("File name captured"):
            self.assertEqual(results['file'], 'test.py')
        with self.subTest("Status updated"):
            self.assertEqual(results['status'], 'Completed')
        # Logs
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM logs WHERE fk = (?)", [system.exp_id])
            results = cursor.fetchall()
        with self.subTest("Log Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("Complete log captured"):
            self.assertListEqual(results['log'].splitlines(), ["Test Log Capture", "Line 2"])
        db.close()

    def test_error_raised(self):
        system = sample_system_object()
        recorder = HistoryRecorder(system=system, est_path="test.py", db_path=self.db_path)
        try:
            with recorder:
                print("Test Log Capture")
                print("Line 2")
                raise RuntimeError("Training Died")
        except RuntimeError:
            pass
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM history WHERE pk = (?)", [system.exp_id])
            results = cursor.fetchall()
        with self.subTest("History Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("File name captured"):
            self.assertEqual(results['file'], 'test.py')
        with self.subTest("Status updated"):
            self.assertEqual(results['status'], 'Failed')
        # Logs
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM logs WHERE fk = (?)", [system.exp_id])
            results = cursor.fetchall()
        with self.subTest("Log Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("Complete log captured"):
            self.assertEqual(results['log'], "Test Log Capture\nLine 2\n")
        # Error
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM errors WHERE fk = (?)", [system.exp_id])
            results = cursor.fetchall()
        with self.subTest("Error Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("Error info captured"):
            self.assertEqual(results['exc_type'], "RuntimeError")
        db.close()

    def test_restore_training(self):
        system1 = sample_system_object()
        recorder1 = HistoryRecorder(system=system1, est_path="test.py", db_path=self.db_path)
        try:
            with recorder1:
                print("Test Log Capture")
                print("Line 2")
                raise RuntimeError("Training Died")
        except RuntimeError:
            pass
        system2 = sample_system_object()
        recorder2 = HistoryRecorder(system=system2, est_path="test.py", db_path=self.db_path)
        with recorder2:
            # Fake a restore wizard
            system2.__dict__.update(system1.__dict__)
            print("Line 3")
            print("Line 4")
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM history")
            results = cursor.fetchall()
        with self.subTest("History Captured and Consolidated"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("File name captured"):
            self.assertEqual(results['file'], 'test.py')
        with self.subTest("Status updated"):
            self.assertEqual(results['status'], 'Completed')
        with self.subTest("Correct PK"):
            self.assertEqual(results['pk'], system1.exp_id)
        with self.subTest("Restarts Incremented"):
            self.assertEqual(results['n_restarts'], 1)
        # Logs
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM logs WHERE fk = (?)", [system1.exp_id])
            results = cursor.fetchall()
        with self.subTest("Log Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("Complete log captured"):
            self.assertListEqual(results['log'].splitlines(), ["Test Log Capture", "Line 2", "Line 3", "Line 4"])
        db.close()

    def test_restore_training_old_missing(self):
        system1 = sample_system_object()
        recorder1 = HistoryRecorder(system=system1, est_path="test.py", db_path=self.db_path)
        try:
            with recorder1:
                print("Test Log Capture")
                print("Line 2")
                raise RuntimeError("Training Died")
        except RuntimeError:
            pass
        db = connect(self.db_path)
        db.execute("DELETE FROM history WHERE pk = (?)", [system1.exp_id])
        db.commit()
        system2 = sample_system_object()
        recorder2 = HistoryRecorder(system=system2, est_path="test.py", db_path=self.db_path)
        with recorder2:
            # Fake a restore wizard
            system2.__dict__.update(system1.__dict__)
            print("Line 3")
            print("Line 4")
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM history")
            results = cursor.fetchall()
        with self.subTest("History Captured and Consolidated"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("File name captured"):
            self.assertEqual(results['file'], 'test.py')
        with self.subTest("Status updated"):
            self.assertEqual(results['status'], 'Completed')
        with self.subTest("Correct PK"):
            self.assertEqual(results['pk'], system1.exp_id)
        with self.subTest("Restarts Incremented"):
            self.assertEqual(results['n_restarts'], 1)
        # Logs
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM logs WHERE fk = (?)", [system1.exp_id])
            results = cursor.fetchall()
        with self.subTest("Log Captured"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("Complete log captured"):
            self.assertListEqual(results['log'].splitlines(), ["Line 3", "Line 4"])
        db.close()

    def test_auto_delete(self):
        update_settings(n_keep=5, db_path=self.db_path)

        for i in range(7):
            system = sample_system_object()
            recorder = HistoryRecorder(system=system, est_path=f"{i}", db_path=self.db_path)
            with recorder:
                print(f"Run {i}")

        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM history")
            results = cursor.fetchall()

        with self.subTest("Ensure correct number retained"):
            self.assertEqual(len(results), 5)
        with self.subTest("Ensure correct entries retained"):
            actual_names = {result['file'] for result in results}
            expected_names = {f"{i}" for i in range(2, 7)}
            self.assertSetEqual(actual_names, expected_names)
        db.close()


class TestHistoryReader(unittest.TestCase):
    def setUp(self):
        self.db_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.db_dir, 'tmp.db')

    def tearDown(self):
        shutil.rmtree(self.db_dir)

    def test_read_basic_with_data(self):
        for i in range(10):
            system = sample_system_object()
            recorder = HistoryRecorder(system=system, est_path=f"{i}", db_path=self.db_path)
            with recorder:
                print("Test Log Capture")
                print("Line 2")
        self.test_read_basic()

    def test_read_basic(self):
        # Ensure none of the options generate invalid queries
        with HistoryReader(db_path=self.db_path) as reader:
            with self.subTest("No Options"):
                reader.read_basic()
            with self.subTest("Limit 20"):
                reader.read_basic(limit=20)
            with self.subTest("CSV"):
                reader.read_basic(as_csv=True)
            with self.subTest("Args"):
                reader.read_basic(include_args=True)
            with self.subTest("Pks"):
                reader.read_basic(include_pk=True)
            # Query Altering
            with self.subTest("Errs"):
                reader.read_basic(errors=True)
            with self.subTest("Features"):
                reader.read_basic(include_features=True)
            with self.subTest("Traces"):
                reader.read_basic(include_traces=True)
            with self.subTest("Datasets"):
                reader.read_basic(include_datasets=True)
            with self.subTest("Pipeline"):
                reader.read_basic(include_pipeline=True)
            with self.subTest("Network"):
                reader.read_basic(include_network=True)
            # Combos
            with self.subTest("Everything"):
                reader.read_basic(errors=True,
                                  include_features=True,
                                  include_traces=True,
                                  include_datasets=True,
                                  include_pipeline=True,
                                  include_network=True)
            with self.subTest("All -Errors"):
                with self.subTest("Everything"):
                    reader.read_basic(errors=False,
                                      include_features=True,
                                      include_traces=True,
                                      include_datasets=True,
                                      include_pipeline=True,
                                      include_network=True)

    def test_read_sql(self):
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                with HistoryReader(db_path=self.db_path) as reader:
                    reader.read_sql(query='SELECT COUNT(*) AS count FROM settings', as_csv=True)
            output = buf.getvalue()
            self.assertRegex(output, '.*count\r\n1\r\n.*')
