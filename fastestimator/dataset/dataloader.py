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
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter


class FEDataLoader(DataLoader):
    _current_threads = []

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
        if isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            FEDataLoader._current_threads.extend([w.pid for w in self._iterator._workers])
        return self._iterator
