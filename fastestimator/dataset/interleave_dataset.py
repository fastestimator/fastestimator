# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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

from typing import List, Mapping, Optional, Union

from torch.utils.data import ConcatDataset, Dataset

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.util.traceability_util import traceable


class InterleaveDataset(FEDataset):
    def __init__(self,
                 datasets: Union[Mapping[str, Dataset], List[Dataset]],
                 pattern: Optional[Union[List[str], List[int]]] = None) -> None:
        print("f")
