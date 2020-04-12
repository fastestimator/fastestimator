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
import os

import pandas as pd

from fastestimator.dataset.dataset import InMemoryDataset


class PickleDataset(InMemoryDataset):
    """A dataset from a pickle file.

    PickleDataset reads entries from pickled pandas data-frames. The root directory of the pickle file may be accessed
    using dataset.parent_path. This may be useful if the file contains relative path information that you want to feed
    into, say, an ImageReader Op.

    Args:
        file_path: The (absolute) path to the pickle file.
    """
    def __init__(self, file_path: str) -> None:
        df = pd.read_pickle(file_path)
        self.parent_path = os.path.dirname(file_path)
        super().__init__(df.to_dict(orient='index'))
