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
from torch.utils.data import Dataset

from fastestimator.util.traceability_util import traceable


@traceable()
class ExtendDataset(Dataset):
    """ExtendDataset either extends or contracts the length of provided Dataset.

    Args:
        dataset: The Original dataset(s) which need expansion or contraction.
        spoof_length: Length to which original dataset must be expanded or contracted to. (New desired length)
    """
    def __init__(self, dataset: Dataset, spoof_length: int) -> None:
        self.dataset = dataset
        self.spoof_length = spoof_length
        self._check_input()
        if hasattr(dataset, "fe_reset_ds"):
            self.fe_reset_ds = dataset.fe_reset_ds
        if hasattr(dataset, "fe_batch_indices"):
            self.fe_batch_indices = dataset.fe_batch_indices
        if hasattr(dataset, "fe_batch"):
            self.fe_batch = dataset.fe_batch

    def __len__(self):
        return len(self.dataset)

    def _check_input(self) -> None:
        """Verify that the given input values are valid.
        Raises:
            AssertionError: If any of the parameters are found to by unacceptable for a variety of reasons.
        """
        assert isinstance(self.spoof_length, int), "Only accept positive integer type as spoof_length"
        assert self.spoof_length > 0, "Invalid spoof_length. Expand Length cannot be less than or equal to 0"
        assert not isinstance(self.dataset, ExtendDataset), "Input Dataset cannot be an ExtendDataset object"

    def __getitem__(self, index):
        return self.dataset[index]
