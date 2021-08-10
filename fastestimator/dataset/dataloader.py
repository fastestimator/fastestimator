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
import re
import signal
import sys
import threading
from signal import Signals
from types import FrameType

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter


class FEDataLoader(DataLoader):
    _SIGCHLD_handler_overridden = False
    _current_threads = []

    def _re_wrap_handler(self) -> None:
        """Intercept the pytorch SIGCHLD handler so that it only worries about its own threads dying.
        """
        if sys.platform == "win32":
            return  # Windows doesn't use this signal handling currently
        if not isinstance(threading.current_thread(), threading._MainThread):  # type: ignore
            return  # Can't set signal in child threads
        if not isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            return  # Only care about multi-processing
        if FEDataLoader._SIGCHLD_handler_overridden:
            return  # Only override signal handler once

        previous_handler = signal.getsignal(signal.SIGCHLD)

        def handler(signum: Signals, frame: FrameType) -> None:
            try:
                previous_handler(signum, frame)
            except RuntimeError as err:
                # Pytorch will raise a runtime error if any thread belonging to any data loader unexpectedly dies at any
                # time. This seems a bit excessive, so we will only raise the error if the dead thread belongs to the
                # current iterator (possible since we use context locking in the pipeline).
                str_err = str(err)
                dead_pid = re.findall("[(]pid ([0-9]+)[)]", str_err)
                if len(dead_pid) == 1:
                    dead_pid = int(dead_pid[0])
                    if dead_pid not in FEDataLoader._current_threads:
                        return  # Don't care if a different data loader has died
                raise err

        signal.signal(signal.SIGCHLD, handler)
        FEDataLoader._SIGCHLD_handler_overridden = True

    def shutdown(self) -> None:
        """Close the worker threads used by this iterator.

        The hope is that this will prevent "RuntimeError: DataLoader worker (pid(s) XXXX) exited unexpectedly" during
        the test suites.
        """
        if isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            self._iterator._shutdown_workers()
        self._iterator = None
        FEDataLoader._current_threads.clear()

    def __iter__(self) -> _BaseDataLoaderIter:
        # Similar to the original iter method, but we remember iterators in order to manually close them when new ones
        # are created
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
        else:
            self.shutdown()
            self._iterator = self._get_iterator()
        self._re_wrap_handler()
        if isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            FEDataLoader._current_threads.extend([w.pid for w in self._iterator._workers])
        return self._iterator
