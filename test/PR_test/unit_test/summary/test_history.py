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
import os
import shutil
import tempfile
import unittest
from contextlib import closing
from datetime import datetime

from fastestimator.summary.history import connect, delete, update_settings

DEFAULT_KEEP = 500
DEFAULT_LOG_KEEP = 500


class TestConnect(unittest.TestCase):
    def test_make_schema(self):
        db = connect(":memory:")
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            results = cursor.fetchall()
        results = {result['name'] for result in results}
        expected = {"datasets", "errors", "features", "history", "pipeline", "network", "postprocess", "traces",
                    "errors", "logs", "settings"}
        self.assertSetEqual(results, expected)
        db.close()

    def test_fk_delete(self):
        db = connect(":memory:")
        db.execute("INSERT INTO history (pk) VALUES (0)")
        db.execute("INSERT INTO network (fk) VALUES (0)")
        db.commit()
        with closing(db.cursor()) as cursor:
            with self.subTest("entry should be inserted correctly"):
                cursor.execute("SELECT count(*) AS count FROM network")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 1)
            with self.subTest("entry should cascade delete"):
                db.execute("DELETE FROM history WHERE pk = 0")
                cursor.execute("SELECT count(*) AS count FROM network")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 0)
        db.close()

    def test_fk_update(self):
        db = connect(":memory:")
        db.execute("INSERT INTO history (pk) VALUES (0)")
        db.execute("INSERT INTO network (fk) VALUES (0)")
        db.commit()
        with closing(db.cursor()) as cursor:
            with self.subTest("entry should be inserted correctly"):
                cursor.execute("SELECT * FROM network")
                results = cursor.fetchall()
                self.assertEqual(results[0]['fk'], 0)
            with self.subTest("entry should cascade update"):
                db.execute("UPDATE history SET pk = 10 WHERE pk = 0")
                cursor.execute("SELECT * FROM network")
                results = cursor.fetchall()
                self.assertEqual(results[0]['fk'], 10)
        db.close()

    def test_initial_settings(self):
        db = connect(":memory:")
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings")
            results = cursor.fetchall()
        with self.subTest("should be exactly one setting row"):
            self.assertEqual(len(results), 1)
        results = results[0]
        with self.subTest("settings should have pk = 0"):
            self.assertEqual(results['pk'], 0)
        with self.subTest("schema version should be 1"):
            self.assertEqual(results['schema_version'], 1)
        with self.subTest(f"n_keep should be {DEFAULT_KEEP}"):
            self.assertEqual(results['n_keep'], DEFAULT_KEEP)
        with self.subTest(f"n_keep_logs should be {DEFAULT_LOG_KEEP}"):
            self.assertEqual(results['n_keep_logs'], DEFAULT_LOG_KEEP)


class TestDelete(unittest.TestCase):
    def setUp(self):
        self.db_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.db_dir, 'tmp.db')

    def tearDown(self):
        shutil.rmtree(self.db_dir)

    def test_under_threshold(self):
        db = connect(self.db_path)
        for i in range(10):
            db.execute("INSERT INTO history (pk, train_start) VALUES (?, ?)", [i, datetime.now()])
        db.commit()
        with closing(db.cursor()) as cursor:
            with self.subTest("entries should be inserted correctly"):
                cursor.execute("SELECT count(*) AS count FROM history")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 10)
            delete(n_keep=10, db_path=self.db_path)
            with self.subTest("no entries should be deleted"):
                cursor.execute("SELECT count(*) AS count FROM history")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 10)
        db.close()

    def test_over_threshold(self):
        db = connect(self.db_path)
        for i in range(10):
            db.execute("INSERT INTO history (pk, train_start) VALUES (?, ?)", [i, datetime.now()])
        db.commit()
        with closing(db.cursor()) as cursor:
            with self.subTest("entries should be inserted correctly"):
                cursor.execute("SELECT count(*) AS count FROM history")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 10)
            delete(n_keep=9, db_path=self.db_path)
            with self.subTest("one entry should be deleted"):
                cursor.execute("SELECT count(*) AS count FROM history")
                results = cursor.fetchall()
                self.assertEqual(results[0]['count'], 9)
            with self.subTest("oldest entry should be deleted"):
                cursor.execute("SELECT pk FROM history")
                results = cursor.fetchall()
                results = {result['pk'] for result in results}
                expected = {i for i in range(1, 10)}
                self.assertSetEqual(results, expected)
        db.close()


class TestUpdateSettings(unittest.TestCase):
    def setUp(self):
        self.db_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.db_dir, 'tmp.db')

    def tearDown(self):
        shutil.rmtree(self.db_dir)

    def test_decrease_n_keep(self):
        update_settings(n_keep=42, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 42)
        with self.subTest("n_keep_logs also reduced"):
            self.assertEqual(results['n_keep_logs'], 42)

    def test_increase_n_keep(self):
        update_settings(n_keep=2000, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 2000)
        with self.subTest("n_keep_logs not updated"):
            self.assertEqual(results['n_keep_logs'], DEFAULT_LOG_KEEP)

    def test_decrease_logs(self):
        update_settings(n_keep_logs=42, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep not updated"):
            self.assertEqual(results['n_keep'], DEFAULT_KEEP)
        with self.subTest("n_keep_logs updated"):
            self.assertEqual(results['n_keep_logs'], 42)

    def test_increase_logs(self):
        update_settings(n_keep_logs=1500, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep not updated"):
            self.assertEqual(results['n_keep'], DEFAULT_KEEP)
        with self.subTest("n_keep_logs not updated (n_logs can't exceed n_keep)"):
            self.assertEqual(results['n_keep_logs'], DEFAULT_LOG_KEEP)

    def test_joint_update_raise_kgl(self):
        update_settings(n_keep=1200, n_keep_logs=1100, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 1200)
        with self.subTest("n_keep_logs updated"):
            self.assertEqual(results['n_keep_logs'], 1100)

    def test_joint_update_raise_lgk(self):
        update_settings(n_keep=1100, n_keep_logs=1200, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 1100)
        with self.subTest("n_keep_logs updated"):
            self.assertEqual(results['n_keep_logs'], 1100)

    def test_joint_update_lower_lgk(self):
        update_settings(n_keep=300, n_keep_logs=400, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 300)
        with self.subTest("n_keep_logs updated"):
            self.assertEqual(results['n_keep_logs'], 300)

    def test_joint_update_lower_kgl(self):
        update_settings(n_keep=400, n_keep_logs=300, db_path=self.db_path)
        db = connect(self.db_path)
        with closing(db.cursor()) as cursor:
            cursor.execute("SELECT * FROM settings WHERE pk = 0")
            results = cursor.fetchall()[0]
        with self.subTest("n_keep updated"):
            self.assertEqual(results['n_keep'], 400)
        with self.subTest("n_keep_logs updated"):
            self.assertEqual(results['n_keep_logs'], 300)
