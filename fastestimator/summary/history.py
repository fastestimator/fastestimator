# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import argparse
import multiprocessing
import os
import sqlite3 as sql
import sys
import traceback
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import tensorflow.keras.mixed_precision as mixed_precision
import torch
from prettytable import PrettyTable, from_db_cursor

from fastestimator.schedule.schedule import Scheduler
from fastestimator.summary.logs.log_parse import parse_log_iter
from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.summary.summary import Summary, average_summaries
from fastestimator.summary.system import System
from fastestimator.util.cli_util import SaveAction
from fastestimator.util.util import NonContext, parse_string_to_python

_MAKE_HIST_TABLE = 'CREATE TABLE IF NOT EXISTS history (' \
                   'file TEXT, ' \
                   'experiment TEXT, ' \
                   'status TEXT, ' \
                   'args LIST[STR], ' \
                   'fe_version TEXT, ' \
                   'train_start TIMESTAMP, ' \
                   'train_end TIMESTAMP, ' \
                   'n_gpus INTEGER, ' \
                   'n_cpus INTEGER, ' \
                   'n_workers INTEGER, ' \
                   'n_restarts INTEGER, ' \
                   'pk INTEGER PRIMARY KEY' \
                   ')'
_MAKE_FEAT_TABLE = 'CREATE TABLE IF NOT EXISTS features (' \
                   'feature TEXT, ' \
                   'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                   ')'
_MAKE_DS_TABLE = 'CREATE TABLE IF NOT EXISTS datasets (' \
                 'mode TEXT, ' \
                 'dataset TEXT, ' \
                 'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                 ')'
_MAKE_PIPELINE_TABLE = 'CREATE TABLE IF NOT EXISTS pipeline (' \
                       'pipe_op TEXT, ' \
                       'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                       ')'
_MAKE_NETWORK_TABLE = 'CREATE TABLE IF NOT EXISTS network (' \
                      'net_op TEXT, ' \
                      'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                      ')'
_MAKE_POST_PROCESS_TABLE = 'CREATE TABLE IF NOT EXISTS postprocess (' \
                           'pp_op TEXT, ' \
                           'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                           ')'
_MAKE_TRACE_TABLE = 'CREATE TABLE IF NOT EXISTS traces (' \
                    'trace TEXT, ' \
                    'fk INTEGER REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                    ')'
_MAKE_ERR_TABLE = 'CREATE TABLE IF NOT EXISTS errors (' \
                  'exc_type TEXT, ' \
                  'exc_tb TEXT, ' \
                  'fk INTEGER PRIMARY KEY REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                  ')'
_MAKE_LOG_TABLE = 'CREATE TABLE IF NOT EXISTS logs (' \
                  'log TEXT, ' \
                  'fk INTEGER PRIMARY KEY REFERENCES history(pk) ON DELETE CASCADE ON UPDATE CASCADE' \
                  ')'
_MAKE_SETTINGS_TABLE = 'CREATE TABLE IF NOT EXISTS settings (' \
                       'pk INTEGER PRIMARY KEY NOT NULL, ' \
                       'schema_version INTEGER, ' \
                       'n_keep INTEGER, ' \
                       'n_keep_logs INTEGER)'

_MAKE_HIST_ENTRY = "INSERT INTO history (" \
                   "pk, file, experiment, args, fe_version, train_start, " \
                   "status, n_gpus, n_cpus, n_workers, n_restarts) " \
                   "VALUES (:pk, :fname, :exp, :args, :version, :start, :status, :gpus, :cpus, :workers, 0)"
_MAKE_FEAT_ENTRY = "INSERT INTO features (feature, fk) VALUES (:feature, :fk)"
_MAKE_DS_ENTRY = "INSERT INTO datasets (mode, dataset, fk) VALUES (:mode, :dataset, :fk)"
_MAKE_PIPE_ENTRY = "INSERT INTO pipeline (pipe_op, fk) VALUES (:op, :fk)"
_MAKE_NET_ENTRY = "INSERT INTO network (net_op, fk) VALUES (:op, :fk)"
_MAKE_PP_ENTRY = "INSERT INTO postprocess (pp_op, fk) VALUES (:op, :fk)"
_MAKE_TRACE_ENTRY = "INSERT INTO traces (trace, fk) VALUES (:op, :fk)"
# It would be nice to use a single INSERT ON CONFLICT call here, but Ubuntu 18 apt-get ships version 3.22.0 of sqlite
# and the new syntax wasn't added until 3.24.0
_MAKE_ERR_ENTRY_P1 = "UPDATE errors SET exc_type = :type, exc_tb = :tb WHERE fk = :fk"
_MAKE_ERR_ENTRY_P2 = "INSERT OR IGNORE INTO errors (exc_type, exc_tb, fk) VALUES (:type, :tb, :fk)"
_MAKE_LOG_ENTRY = "INSERT INTO logs (log, fk) VALUES (:log, :fk)"
_MAKE_SETTINGS_ENTRY = "INSERT OR IGNORE INTO settings (pk, schema_version, n_keep, n_keep_logs) " \
                       "VALUES (0, 1, 500, 500)"


def connect(db_path: Optional[str] = None) -> sql.Connection:
    """Open a connection to a sqlite database, creating one if it does not already exist.

    Args:
        db_path: The path to the database file. Or None to default to ~/fastestimator_data/history.db

    Returns:
        An open connection to the database, with schema instantiated and foreign keys enabled.
    """
    if db_path is None:
        db_path = os.path.join(str(Path.home()), 'fastestimator_data', 'history.db')
    if db_path != ':memory:':  # This is a reserved keyword to create an in-memory database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Make sure folders exist before creating disk file
    connection = sql.connect(db_path, detect_types=sql.PARSE_DECLTYPES | sql.PARSE_COLNAMES)
    connection.execute("PRAGMA foreign_keys = 1")  # Enable FK constraints
    connection.row_factory = sql.Row  # Get nice query return objects
    # Build the schema if it doesn't exist
    connection.execute(_MAKE_HIST_TABLE)
    connection.execute(_MAKE_FEAT_TABLE)
    connection.execute(_MAKE_DS_TABLE)
    connection.execute(_MAKE_PIPELINE_TABLE)
    connection.execute(_MAKE_NETWORK_TABLE)
    connection.execute(_MAKE_POST_PROCESS_TABLE)
    connection.execute(_MAKE_TRACE_TABLE)
    connection.execute(_MAKE_ERR_TABLE)
    connection.execute(_MAKE_LOG_TABLE)
    connection.execute(_MAKE_SETTINGS_TABLE)
    connection.execute(_MAKE_SETTINGS_ENTRY)
    connection.commit()
    return connection


def delete(n_keep: int = 20, db_path: Optional[str] = None) -> None:
    """Remove history entries from a database.

    This will also remove associated data such as logs due to foreign key constraints.

    Args:
        n_keep: How many history entries to keep.
        db_path: The path to the database, or None to use the default location.
    """
    db = connect(db_path)
    db.execute(
        "DELETE FROM history WHERE train_start <= ("
        "SELECT train_start FROM history ORDER BY train_start DESC LIMIT 1 OFFSET (?))", [n_keep])
    db.commit()  # Can't vacuum while there are uncommitted changes
    db.execute("VACUUM")  # Free the memory
    db.commit()
    db.close()


def update_settings(n_keep: Optional[int] = None, n_keep_logs: Optional[int] = None,
                    db_path: Optional[str] = None) -> None:
    """Update the history database settings.

    Updated settings will be enforced the next time a training or delete operation is called.

    Args:
        n_keep: How many history entries should be retained.
        n_keep_logs: How many logs should be retained. This value should be <= `n_keep`.
        db_path: The path to the database, or None to use the default location.
    """
    db = connect(db_path)
    # Ensure limits are non-negative
    if n_keep:
        n_keep = max(n_keep, 0)
    if n_keep_logs:
        n_keep_logs = max(n_keep_logs, 0)
    # Perform the update
    if n_keep is not None and n_keep_logs is not None:
        db.execute("UPDATE settings SET n_keep = :keep, n_keep_logs = MIN(:keep, :logs) WHERE pk = 0", {
            'keep': n_keep, 'logs': n_keep_logs
        })
    elif n_keep is not None:
        db.execute("UPDATE settings SET n_keep = :keep, n_keep_logs = MIN(n_keep_logs, :keep) WHERE pk = 0",
                   {'keep': n_keep})
    elif n_keep_logs is not None:
        db.execute("UPDATE settings SET n_keep_logs = MIN(n_keep, (?)) WHERE pk = 0", [n_keep_logs])
    db.commit()
    with closing(db.cursor()) as cursor:
        cursor.execute("SELECT * FROM settings")
        response = from_db_cursor(cursor)
    # Hide implementation details from end user
    response.del_column('pk')
    response.del_column('schema_version')
    print(response)
    db.close()


class HistoryRecorder:
    """A class to record what you're doing.

    This class is intentionally not @traceable.

    It will capture output logs, exceptions, and general information about the training / environment. This class should
    be used as a context manager.

    Args:
        system: The system object corresponding to the current training.
        est_path: The path to the file responsible for creating the current estimator (this is for bookkeeping, it can
            technically be any string).
        db_path: The path to the database, or None to use the default location.
    """
    def __init__(self, system: System, est_path: str, db_path: Optional[str] = None):
        # Prepare db adapters
        sql.register_adapter(bool, int)
        sql.register_converter("BOOL", lambda v: bool(int(v)))
        sql.register_adapter(list, str)
        sql.register_converter("LIST[STR]", lambda v: parse_string_to_python(v))
        # Prepare variables
        self.filename = os.path.basename(est_path)
        self.db_path = db_path if db_path else os.path.join(str(Path.home()), 'fastestimator_data', 'history.db')
        self.system = system
        self.db = None
        self.pk = None
        self.stdout = None

    def __enter__(self) -> None:
        self.db = connect(self.db_path)
        self.pk = self.system.exp_id  # This might be changed later by RestoreWizard. See the _check_for_restart method
        # Check whether an entry for this pk already exists, for example if a user ran .fit() and is now running .test()
        with closing(self.db.cursor()) as cursor:
            exists = cursor.execute("SELECT pk FROM history WHERE pk = (?)", [self.pk])
            exists = exists.fetchall()
        if not exists:
            self.db.execute(
                _MAKE_HIST_ENTRY,
                {
                    'pk': self.pk,
                    'fname': self.filename,
                    'status': 'Launched',
                    'exp': self.system.summary.name,
                    'args': sys.argv[1:],
                    'version': sys.modules['fastestimator'].__version__,
                    'start': datetime.now(),
                    'gpus': torch.cuda.device_count(),
                    'cpus': os.cpu_count(),
                    'workers': self.system.pipeline.num_process
                })
            self.db.executemany(_MAKE_FEAT_ENTRY, self._get_features_in_use())
            self.db.executemany(_MAKE_DS_ENTRY, self._get_datasets_in_use())
            self.db.executemany(_MAKE_PIPE_ENTRY, self._get_items_in_use(self.system.pipeline.ops))
            self.db.executemany(_MAKE_NET_ENTRY, self._get_items_in_use(self.system.network.ops))
            self.db.executemany(_MAKE_PP_ENTRY, self._get_items_in_use(self.system.network.postprocessing))
            self.db.executemany(_MAKE_TRACE_ENTRY, self._get_items_in_use(self.system.traces))
            self.db.execute(_MAKE_LOG_ENTRY, {'log': '', 'fk': self.pk})
            self.db.commit()
        # Take over the output logging
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type: Optional[Type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self._check_for_restart()
        self.flush()
        sys.stdout = self.stdout
        # In test mode only overwrite the train_end time if it hasn't already been set
        train_end_query = "(?)" if self.system.mode in ('train', 'eval') else "IFNULL(train_end, (?))"
        query = f"UPDATE history set train_end = {train_end_query}, status = (?) WHERE pk = (?)"
        self.db.execute(
            query,
            [
                datetime.now(),
                "Completed" if exc_type is None else "Aborted" if exc_type == KeyboardInterrupt else "Failed",
                self.pk
            ])
        if exc_type is not None:
            args = {
                'type': exc_type.__name__,
                'tb': "\n".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
                'fk': self.pk
            }
            self.db.execute(_MAKE_ERR_ENTRY_P1, args)
            self.db.execute(_MAKE_ERR_ENTRY_P2, args)
        self.db.commit()
        self._apply_limits()
        self.db.close()

    def _check_for_restart(self) -> None:
        """Determine whether a training has been restarted via RestoreWizard. If so, update the history accordingly.

        If RestoreWizard has been invoked, then the system exp_id will have changed since self.pk was initialized. This
        method will do related bookkeeping, and then swap self.pk for the restored id.
        """
        if self.pk == self.system.exp_id:
            return
        # RestoreWizard reset the system, we are continuing an old training rather than starting a new one
        # First make sure the old entry is still available to edit
        with closing(self.db.cursor()) as cursor:
            exists = cursor.execute("SELECT pk FROM history WHERE pk = (?)", [self.system.exp_id])
            exists = exists.fetchall()
        if exists:
            # If we still have the original entry, we will delete our new one and update the old instead
            self.db.execute("DELETE FROM history WHERE pk = (?)", [self.pk])
        else:
            # The old record doesn't exist, so we will use the new record instead
            self.db.execute("UPDATE history SET pk = (?) WHERE pk = (?)", [self.system.exp_id, self.pk])
        self.pk = self.system.exp_id
        self.db.execute("UPDATE history SET n_restarts = n_restarts + 1 WHERE pk = (?)", [self.pk])
        self.db.commit()

    def _get_features_in_use(self) -> List[Dict[str, str]]:
        """Determine which interesting FE features are being used by the current training.

        Returns:
            A list of entries which can be written into the 'features' db table.
        """
        features = []
        if sys.modules['fastestimator'].fe_deterministic_seed is not None:
            features.append({'feature': 'Deterministic', 'fk': self.pk})
        if any([len(mode_dict) > 1 for mode_dict in self.system.pipeline.data.values()]):
            features.append({'feature': 'MultiDataset', 'fk': self.pk})
        if mixed_precision.global_policy().compute_dtype == 'float16':
            features.append({'feature': 'MixedPrecision', 'fk': self.pk})
        return features

    def _get_datasets_in_use(self) -> List[Dict[str, str]]:
        """Determine which datasets are being used by the current training.

        Returns:
            A list of entries which can be written into the 'datasets' db table.
        """
        datasets = []
        for mode, group in self.system.pipeline.data.items():
            for _, ds in group.items():
                datasets.append({'mode': mode, 'dataset': type(ds).__name__, 'fk': self.pk})
        return datasets

    def _get_items_in_use(self, items: List[Any]) -> List[Dict[str, str]]:
        """Determine which objects are being used by the current training.

        Args:
            items: A list of Schedulers, Ops, and/or traces which are being used by the system.

        Returns:
            The elements from `items` converted into database-ready entries.
        """
        ops = []
        for op in items:
            op_list = [op]
            if isinstance(op, Scheduler):
                op_list = list(filter(lambda x: x is not None, op.get_all_values()))
                op_list.append(op)  # Put scheduler in too so that usage can be tracked too
            ops.extend([{'op': type(elem).__name__, 'fk': self.pk} for elem in op_list])
        return ops

    def _apply_limits(self) -> None:
        """Remove old history and/or log entries if they exceed the limits defined in the settings table.
        """
        self.db.execute("DELETE FROM history WHERE train_start <= ("
                        "SELECT train_start FROM history ORDER BY train_start DESC LIMIT 1 OFFSET ("
                        "SELECT n_keep FROM settings WHERE pk = 0))")
        self.db.execute("DELETE FROM logs WHERE fk IN ("
                        "SELECT pk FROM history ORDER BY train_start DESC LIMIT 1 OFFSET ("
                        "SELECT n_keep_logs FROM settings WHERE pk = 0))")
        self.db.commit()  # Have to commit before vacuuming
        if sum(int(digit) for digit in str(abs(self.pk))) % 10 == 0:
            # 10% of time do a vacuum (expensive). We don't use random.randint here due to deterministic training. Also,
            # don't use pk directly because last digit is not uniformly distributed.
            self.db.execute("PRAGMA VACUUM;")
            self.db.commit()
        else:
            # Otherwise do a less costly optimize
            self.db.execute("PRAGMA optimize;")
            self.db.commit()
        self.db.close()

    def write(self, output: str) -> None:
        self._check_for_restart()  # Check here instead of just waiting for __exit__ in case system powers off later
        self.stdout.write(output)
        if multiprocessing.current_process().name == 'MainProcess':
            # Flush can also get invoked by pipeline multi-processing, but db should only be accessed by main thread.
            # This can happen, for example, when pipeline prints a warning that a certain key is unused and will be
            # dropped.
            self.db.execute('UPDATE logs SET log = log || (?) WHERE fk = (?)', [output, self.pk])
            self.db.commit()

    def flush(self) -> None:
        self.stdout.flush()


class HistoryReader:
    """A class to read history information from the database.

    This class is intentionally not @traceable.

    This class should be used as as a context manager, for example:

    ```python
    with HistoryReader() as reader:
        reader.read_basic()
    ```

    Args:
        db_path: The path to the database, or None to use the default location.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.db = None  # sql.Connection
        self.response = None  # List[sql.Row]

    def __enter__(self) -> 'HistoryReader':
        self.db = connect(self.db_path)
        self.db.set_trace_callback(print)  # Show the query in case user wants to adapt it later
        return self

    def __exit__(self, exc_type: Optional[Type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.db.close()

    def read_basic(self,
                   limit: int = 10,
                   interactive: bool = False,
                   include_args: bool = False,
                   errors: bool = False,
                   include_pk: bool = False,
                   include_features: bool = False,
                   include_traces: bool = False,
                   include_datasets: bool = False,
                   include_pipeline: bool = False,
                   include_network: bool = False,
                   as_csv: bool = False) -> None:
        """Perform a pre-defined (and possibly interactive) set of sql selects against the history database.

        Outputs will be printed to stdout.

        Args:
            limit: The maximum number of responses to look up.
            interactive: Whether to run this function interactively, prompting the user for additional input along the
                way. This enables things like error and log retrieval for individual experiments.
            include_args: Whether to output the arguments used to run each experiment.
            errors: Whether to filter the output to only include failed experiments, as well as including more
                information about the specific errors that occurred.
            include_pk: Whether to include the primary keys (experiment ids) of each history entry.
            include_features: Whether to include the FE features that were employed by each training.
            include_traces: Whether to include the traces that were used in each training.
            include_datasets: Whether to include the dataset (classes) that were used in each training.
            include_pipeline: Whether to include the pipeline ops that were used in each training.
            include_network: Whether to include the network (post)processing ops that were used in each training.
            as_csv: Whether to print the output as a csv rather than in a formatted table.
        """
        # Build the query string
        error_select = ", errors.exc_type error" if errors else ''
        error_join = "LEFT JOIN errors ON errors.fk = h.pk " if errors else ''
        error_where = " WHERE h.status <> 'Completed' " if errors else ''
        feature_select = ", fg.features" if include_features else ''
        feature_join = "LEFT JOIN (" \
                       "SELECT fk, GROUP_CONCAT(feature, ', ') features FROM features f GROUP BY f.fk" \
                       ") fg ON fg.fk = h.pk " if include_features else ''
        dataset_select = ", dsg.datasets " if include_datasets else ''
        dataset_join = "LEFT JOIN (" \
                       "SELECT fk, GROUP_CONCAT(dataset || ' (' || mode || ')', ', ') datasets " \
                       "FROM datasets ds GROUP BY ds.fk" \
                       ") dsg ON dsg.fk = h.pk " if include_datasets else ''
        pipeline_select = ", pg.pipeline_ops" if include_pipeline else ''
        pipeline_join = "LEFT JOIN (" \
                        "SELECT fk, GROUP_CONCAT(pipe_op, ', ') pipeline_ops FROM pipeline p GROUP BY p.fk" \
                        ") pg ON pg.fk = h.pk " if include_pipeline else ''
        network_select = ", ng.network_ops, ppg.postprocessing_ops" if include_network else ''
        network_join = "LEFT JOIN (" \
                       "SELECT fk, GROUP_CONCAT(net_op, ', ') network_ops FROM network n GROUP BY n.fk" \
                       ") ng ON ng.fk = h.pk " \
                       "LEFT JOIN (" \
                       "SELECT fk, GROUP_CONCAT(pp_op, ', ') postprocessing_ops FROM postprocess pp GROUP BY pp.fk" \
                       ") ppg ON ppg.fk = h.pk " if include_network else ''
        trace_select = ", tg.traces " if include_traces else ''
        trace_join = "LEFT JOIN (" \
                     "SELECT fk, GROUP_CONCAT(trace, ', ') traces FROM traces t GROUP BY t.fk" \
                     ") tg ON tg.fk = h.pk " if include_traces else ''
        query = f"SELECT h.*{error_select}{feature_select}{dataset_select}{pipeline_select}{network_select}" \
                f"{trace_select} FROM history h {error_join}{feature_join}{dataset_join}{pipeline_join}{network_join}" \
                f"{trace_join}{error_where}ORDER BY h.train_start DESC LIMIT (?)"
        # We have to hide these after-the-fact since later process may require pk behind the scenes
        hide = []
        if not include_pk:
            hide.append('pk')
        if not include_args:
            hide.append('args')
        self.read_sql(query, args=[limit], hide_cols=hide, as_csv=as_csv, interactive=interactive)

    def read_sql(self,
                 query: str,
                 args: Iterable[Any] = (),
                 hide_cols: Iterable[str] = (),
                 as_csv: bool = False,
                 interactive: bool = False) -> None:
        """Perform a (possibly interactive) sql query against the database.

        Args:
            query: The sql query to execute.
            args: Any parameterized arguments to be inserted into the `query`.
            hide_cols: Any columns to hide from the printed output.
            as_csv: Whether to print the output in csv format or in table format.
            interactive: Whether to run this function interactively, prompting the user for additional input along the
                way. This enables things like error and log retrieval for individual experiments.
        """
        with closing(self.db.cursor()) as cursor:
            cursor.execute(query, args)
            self.response = cursor.fetchall()
            names = [col[0] for col in cursor.description]
        # Build nice output table
        table = PrettyTable()
        table.field_names = names
        for row in self.response:
            table.add_row(row)
        for col in hide_cols:
            if col in table.field_names:
                table.del_column(col)
        if interactive:
            table.add_autoindex()
        if as_csv:
            print(table.get_csv_string())
        else:
            print(table)
        if interactive:
            while True:
                inp = input("\033[93m{}\033[00m".format("Enter --help for available command details. Enter without an "
                                                        "argument to re-print the current response. X to exit.\n"))
                if inp in ('X', 'x'):
                    break
                if inp == "":
                    print(query)
                    print(table)
                    continue
                new_query = self._parse_input(inp)
                if new_query:
                    return self.read_sql(new_query, hide_cols=hide_cols, as_csv=as_csv, interactive=interactive)

    def _parse_input(self, inp: str) -> Optional[str]:
        """Take cli input and run it through command parsers to execute an appropriate subroutine.

        Args:
            inp: The cli input provided by an end user.

        Returns:
            The output (if any) of the appropriate sub-command after executing on the given input.
        """
        parser = argparse.ArgumentParser(allow_abbrev=False)
        subparsers = parser.add_subparsers()
        subparsers.required = True
        subparsers.dest = 'cmd'
        self._configure_sql_parser(subparsers)
        self._configure_log_parser(subparsers)
        self._configure_err_parser(subparsers)
        self._configure_vis_parser(subparsers)
        try:
            args, unknown = parser.parse_known_args(inp.split())
        except SystemExit:
            return
        return args.func(vars(args), unknown)

    def _configure_sql_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add a sql parser to an existing argparser.

        Args:
            subparsers: The parser object to be appended to.
        """
        p_sql = subparsers.add_parser('sql',
                                      description='Provide a new sql query to be executed',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      allow_abbrev=False)
        p_sql.add_argument('query', metavar='<Query>', type=str, nargs='+', help="ex: sql SELECT * FROM history")
        p_sql.set_defaults(func=self._echo_sql)

    def _configure_log_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add a log parser to an existing argparser.

        Args:
            subparsers: The parser object to be appended to.
        """
        p_log = subparsers.add_parser(
            'log',
            description='Retrieve one or more output logs. This command requires '
            'that you currently have experiments selected.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False)
        p_log.add_argument('indices',
                           metavar='I',
                           type=int,
                           nargs='+',
                           help="Indices of experiments for which to print logs")
        p_log.add_argument(
            '--file',
            metavar='F',
            action=SaveAction,
            default=False,
            dest='file',
            nargs='?',
            help='Whether to write the logs to disk. May be accompanied by a directory or filename into which to save \
                 the log(s). If none is specified then the ~/fastestimator_data directory will be used.')
        p_log.set_defaults(func=self._fetch_logs)

    def _configure_err_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add an error parser to an existing argparser.

        Args:
            subparsers: The parser object to be appended to.
        """
        p_err = subparsers.add_parser(
            'err',
            description='Retrieve one or more error tracebacks. This command requires '
            'that you currently have experiments selected.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False)
        p_err.add_argument('indices',
                           metavar='I',
                           type=int,
                           nargs='+',
                           help="Indices of experiments for which to print error tracebacks")
        p_err.set_defaults(func=self._fetch_errs)

    def _configure_vis_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add a visualization parser to an existing argparser.

        Args:
            subparsers: The parser object to be appended to.
        """
        p_vis = subparsers.add_parser(
            'vis',
            description='Visualize logs for one or more experiments. This command requires '
            'that you currently have experiments selected.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False)
        p_vis.add_argument('indices',
                           metavar='idx',
                           type=int,
                           nargs='*',
                           help="Indices of experiments for which to print logs")
        group = p_vis.add_mutually_exclusive_group()
        group.add_argument('--ignore',
                           metavar='I',
                           type=str,
                           nargs='+',
                           help="The names of metrics to ignore though they may be present in the log files")
        group.add_argument('--include',
                           metavar='Y',
                           type=str,
                           nargs='+',
                           help="The names of metrics to include. If provided, any other metrics will be ignored.")
        p_vis.add_argument('--smooth',
                           metavar='<float>',
                           type=float,
                           help="The amount of gaussian smoothing to apply (zero for no smoothing)",
                           default=1)
        p_vis.add_argument('--pretty_names', help="Clean up the metric names for display", action='store_true')
        p_vis.add_argument('-g', '--group', dest='groups', default={}, action=_GroupAction, nargs="*")

        legend_group = p_vis.add_argument_group('legend arguments')
        legend_x_group = legend_group.add_mutually_exclusive_group(required=False)
        legend_x_group.add_argument('--common_legend',
                                    dest='share_legend',
                                    help="Generate one legend total",
                                    action='store_true',
                                    default=True)
        legend_x_group.add_argument('--split_legend',
                                    dest='share_legend',
                                    help="Generate one legend per graph",
                                    action='store_false',
                                    default=False)

        save_group = p_vis.add_argument_group('output arguments')
        save_x_group = save_group.add_mutually_exclusive_group(required=False)
        save_x_group.add_argument(
            '--save',
            nargs='?',
            metavar='<Save Dir>',
            dest='save',
            action=SaveAction,
            default=False,
            help="Save the output image. May be accompanied by a directory into \
                  which the file is saved. If no output directory is specified, the history directory will be used")
        save_x_group.add_argument('--display',
                                  dest='save',
                                  action='store_false',
                                  help="Render the image to the UI (rather than saving it)",
                                  default=True)
        save_x_group.set_defaults(save_dir=None)
        p_vis.set_defaults(func=self._vis_logs)

    @staticmethod
    def _echo_sql(args: Dict[str, Any], unknown: List[str]) -> Optional[str]:
        """A method to compile parsed user input back into a single sql query.

        Args:
            args: The CLI arguments provided by the user.
            unknown: Any CLI arguments not matching known inputs.

        Returns:
            A single string containing the user sql query.
        """
        if len(unknown) > 0:
            print("unrecognized arguments: ", str.join(", ", unknown))
            return None
        return " ".join(args['query'])

    def _fetch_logs(self, args: Dict[str, Any], unknown: List[str]) -> None:
        """A method to collect and return a given set of logs from the database.

        Args:
            args: The CLI arguments provided by the user.
            unknown: Any CLI arguments not matching known inputs.
        """
        if len(unknown) > 0:
            print("unrecognized arguments: ", str.join(", ", unknown))
            return
        save = args['file']
        save_path = None
        if save:
            save_path = args['file_dir']
            if save_path is None:
                save_path = os.path.join(str(Path.home()), 'fastestimator_data')
                save = 'dir'
                print(f"Writing log(s) to {save_path}")
            else:
                save = 'file'
                print(f'Writing log to {save_path}')
        logs = {}
        for idx in args['indices']:
            selection = self.response[idx - 1]  # Auto index starts at 1
            pk = selection['pk']
            with closing(self.db.cursor()) as cursor:
                cursor.execute("SELECT log FROM logs WHERE logs.fk = (?)", [pk])
                logs[idx] = cursor.fetchall()
        with open(save_path, 'w') if save == 'file' else NonContext() as f:
            f = sys.stdout if f is None else f
            for idx, log in logs.items():
                with open(os.path.join(save_path, f"{idx}.txt"), 'w') if save == 'dir' else NonContext() as f1:
                    f1 = f if f1 is None else f1
                    if log:
                        f1.write(f'\n@@@@@@@@@@@ Log for Index {idx} @@@@@@@@@@@\n\n')
                        f1.write(log[0]['log'])
                        f1.write('\n')
                    else:
                        f1.write(f"No logs found for Index {idx}\n")

    def _fetch_errs(self, args: Dict[str, Any], unknown: List[str]) -> None:
        """A method to collect and return a given set of error logs from the database.

        Args:
            args: The CLI arguments provided by the user.
            unknown: Any CLI arguments not matching known inputs.
        """
        if len(unknown) > 0:
            print("unrecognized arguments: ", str.join(", ", unknown))
            return
        for idx in args['indices']:
            selection = self.response[idx - 1]  # Auto index starts at 1
            pk = selection['pk']
            with closing(self.db.cursor()) as cursor:
                cursor.execute("SELECT exc_tb FROM errors WHERE errors.fk = (?)", [pk])
                err = cursor.fetchall()
            if err:
                print(f'@@@@@@@@@@@ Traceback for Index {idx} @@@@@@@@@@@')
                print(err[0]['exc_tb'])
            else:
                print(f"No error traceback found for Index {idx}")

    def _vis_logs(self, args: Dict[str, Any], unknown: List[str]) -> None:
        """A method to collect and visualize a given set of logs from the database.

        Args:
            args: The CLI arguments provided by the user.
            unknown: Any CLI arguments not matching known inputs.
        """
        if len(unknown) > 0:
            print("unrecognized arguments: ", str.join(", ", unknown))
            return
        save_dir = args['save_dir']
        if args['save'] and save_dir is None:
            save_dir = os.path.join(str(Path.home()), 'fastestimator_data', 'logs.png')
        group_indices = [x for y in args['groups'].values() for x in y]
        pks = {idx: self.response[idx - 1]['pk'] for idx in set(args['indices'] + group_indices)}
        if len(pks) == 0:
            return
        with closing(self.db.cursor()) as cursor:
            cursor.execute(
                "SELECT H.pk, H.experiment, L.log "
                "FROM logs L LEFT JOIN history H ON L.fk = H.pk "
                "WHERE L.fk IN ({seq})".format(seq=','.join(['?'] * len(pks))),
                list(pks.values()))
            logs = cursor.fetchall()
        logs = {elem['pk']: elem for elem in logs}
        failures = 0
        for idx, pk in pks.items():
            if pk not in logs:
                print(f"No logs found for Index {idx}")
                failures += 1
        if failures:
            return
        groups = defaultdict(list)  # {group_name: [experiment(s)]}
        for idx, pk in pks.items():
            log = logs[pk]
            experiment = parse_log_iter(
                log['log'].split('\n'),
                Summary(str(idx) if log['experiment'] is None else f"{log['experiment']} ({idx})"))
            if idx in args['indices']:
                groups[experiment.name].append(experiment)
            for group_name, group_indices in args['groups'].items():
                if idx in group_indices:
                    groups[group_name].append(experiment)
        experiments = [average_summaries(name, exps) for name, exps in groups.items()]
        visualize_logs(experiments,
                       save_path=save_dir,
                       smooth_factor=args['smooth'],
                       share_legend=args['share_legend'],
                       pretty_names=args['pretty_names'],
                       ignore_metrics=args['ignore'],
                       include_metrics=args['include'])


class _GroupAction(argparse._AppendAction):
    """An argparse action which can be invoked multiple times in order to build a dictionary of entries.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            raise argparse.ArgumentError(self, "--group arguments should of the form <name> <idx1> [<idx2> ...]")
        key = values[0]
        indices = [int(val) for val in values[1:]]
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = {}
        items = items.copy()
        items[key] = indices
        setattr(namespace, self.dest, items)
