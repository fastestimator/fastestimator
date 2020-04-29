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
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

from fastestimator.util.util import to_set

T = TypeVar('T')


class Scheduler(Generic[T]):
    """A class which can wrap things like Datasets and Ops to make their behavior epoch-dependent.
    """
    def get_current_value(self, epoch: int) -> Optional[T]:
        """Fetch whichever of the `Scheduler`s elements is appropriate based on the current epoch.

        Args:
            epoch: The current epoch.

        Returns:
            The element from the Scheduler to be used at the given `epoch`. This value might be None.
        """
        raise NotImplementedError

    def get_all_values(self) -> List[Optional[T]]:
        """Get a list of all the possible values stored in the `Scheduler`.

        Returns:
            A list of all the values stored in the `Scheduler`. This may contain None values.
        """
        raise NotImplementedError


class RepeatScheduler(Scheduler[T]):
    """A scheduler which repeats a collection of entries one after another every epoch.

    One case where this class would be useful is if you want to perform one version of an Op on even epochs, and a
    different version on odd epochs. None values can be used to achieve an end result of skipping an Op every so often.

    ```python
    s = fe.schedule.RepeatScheduler(["a", "b", "c"])
    s.get_current_value(epoch=1)  # "a"
    s.get_current_value(epoch=2)  # "b"
    s.get_current_value(epoch=3)  # "c"
    s.get_current_value(epoch=4)  # "a"
    s.get_current_value(epoch=5)  # "b"
    ```

    Args:
        repeat_list: What elements to cycle between every epoch. Note that epochs start counting from 1. To have nothing
        happen for a particular epoch, None values may be used.

    Raises:
        AssertionError: If `repeat_list` is not a List.
    """
    def __init__(self, repeat_list: List[Optional[T]]) -> None:
        assert isinstance(repeat_list, List), "must provide a list as input of RepeatSchedule"
        self.repeat_list = repeat_list
        self.cycle_length = len(repeat_list)
        assert self.cycle_length > 1, "list length must be greater than 1"

    def get_current_value(self, epoch: int) -> Optional[T]:
        # epoch-1 since the training epoch is 1-indexed rather than 0-indexed.
        return self.repeat_list[(epoch - 1) % self.cycle_length]

    def get_all_values(self) -> List[Optional[T]]:
        return self.repeat_list


class EpochScheduler(Scheduler[T]):
    """A scheduler which selects entries based on a specified epoch mapping.

    This can be useful for making networks grow over time, or to use more challenging data augmentation as training
    progresses.

    ```python
    s = fe.schedule.EpochScheduler({1:"a", 3:"b", 4:None, 100: "c"})
    s.get_current_value(epoch=1)  # "a"
    s.get_current_value(epoch=2)  # "a"
    s.get_current_value(epoch=3)  # "b"
    s.get_current_value(epoch=4)  # None
    s.get_current_value(epoch=99)  # None
    s.get_current_value(epoch=100)  # "c"
    ```

    Args:
        epoch_dict: A mapping from epoch -> element. For epochs in between keys in the dictionary, the closest prior key
            will be used to determine which element to return. None values may be used to cause nothing to happen for a
            particular epoch.

    Raises:
        AssertionError: If the `epoch_dict` is of the wrong type, or contains invalid keys.
    """
    def __init__(self, epoch_dict: Dict[int, T]) -> None:
        assert isinstance(epoch_dict, dict), "must provide dictionary as epoch_dict"
        self.epoch_dict = epoch_dict
        self.keys = sorted(self.epoch_dict)
        for key in self.keys:
            assert isinstance(key, int), "found non-integer key: {}".format(key)
            assert key >= 1, "found non-positive key: {}".format(key)

    def get_current_value(self, epoch: int) -> Optional[T]:
        if epoch in self.keys:
            value = self.epoch_dict[epoch]
        else:
            last_key = self._get_last_key(epoch)
            if last_key is None:
                value = None
            else:
                value = self.epoch_dict[last_key]
        return value

    def get_all_values(self) -> List[Optional[T]]:
        return list(self.epoch_dict.values())

    def _get_last_key(self, epoch: int) -> Union[int, None]:
        """Find the nearest prior key to the given epoch.

        Args:
            epoch: The current target epoch.

        Returns:
            The largest epoch number <= the given `epoch` that is in the `epoch_dict`.
        """
        last_key = None
        for key in self.keys:
            if key > epoch:
                break
            last_key = key
        return last_key


def get_signature_epochs(items: List[Any], total_epochs: int, mode: Optional[str] = None) -> List[int]:
    """Find all epochs of changes due to schedulers.

    Args:
        items: List of items to scan from.
        total_epochs: The maximum epoch number to consider when searching for signature epochs.
        mode: Current execution mode. If None, all execution modes will be considered.

    Returns:
        The epoch numbers of changes.
    """
    unique_configs = []
    signature_epochs = []
    for epoch in range(1, total_epochs + 1):
        epoch_config = get_current_items(items, run_modes=mode, epoch=epoch)
        if epoch_config not in unique_configs:
            unique_configs.append(epoch_config)
            signature_epochs.append(epoch)
    return signature_epochs


def get_current_items(items: Iterable[Union[T, Scheduler[T]]],
                      run_modes: Optional[Union[str, Iterable[str]]] = None,
                      epoch: Optional[int] = None) -> List[T]:
    """Select items which should be executed for given mode and epoch.

    Args:
        items: A list of possible items or Schedulers of items to choose from.
        run_modes: The desired execution mode. One or more of "train", "eval", "test", or "infer". If None, items of
            all modes will be returned.
        epoch: The desired execution epoch. If None, items across all epochs will be returned.

    Returns:
        The items which should be executed.
    """
    selected_items = []
    run_modes = to_set(run_modes)
    for item in items:
        if isinstance(item, Scheduler):
            if epoch is None:
                item = item.get_all_values()
            else:
                item = [item.get_current_value(epoch)]
        else:
            item = [item]
        for item_ in item:
            if item_ and (not run_modes or not hasattr(item_, "mode") or not item_.mode
                          or item_.mode.intersection(run_modes)):
                selected_items.append(item_)
    return selected_items
