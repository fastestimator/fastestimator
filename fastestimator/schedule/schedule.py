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
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union, overload

from fastestimator.util.base_util import to_set
from fastestimator.util.traceability_util import is_restorable, traceable

T = TypeVar('T')
T2 = TypeVar('T2')


@traceable()
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


@traceable()
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

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {
            'repeat_list': [
                elem if is_restorable(elem)[0] else elem.__getstate__() if hasattr(elem, '__getstate__') else {}
                for elem in self.repeat_list
            ]
        }


@traceable()
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

    def __getstate__(self) -> Dict[str, Dict[int, Dict[Any, Any]]]:
        return {
            'epoch_dict': {
                key: elem if is_restorable(elem)[0] else elem.__getstate__() if hasattr(elem, '__getstate__') else {}
                for key,
                elem in self.epoch_dict.items()
            }
        }


def get_signature_epochs(items: List[Any], total_epochs: int, mode: Optional[str] = None, ds_id: Optional[str] = None) \
        -> List[int]:
    """Find all epochs of changes due to schedulers.

    Args:
        items: List of items to scan from.
        total_epochs: The maximum epoch number to consider when searching for signature epochs.
        mode: Current execution mode. If None, all execution modes will be considered.
        ds_id: Current ds_id. If None, all ds_ids will be considered.

    Returns:
        The epoch numbers of changes.
    """
    unique_configs = []
    signature_epochs = []
    for epoch in range(1, total_epochs + 1):
        epoch_config = get_current_items(items, run_modes=mode, epoch=epoch, ds_id=ds_id)
        if epoch_config not in unique_configs:
            unique_configs.append(epoch_config)
            signature_epochs.append(epoch)
    return signature_epochs


@overload
def get_current_items(items: Iterable[Union[T, Scheduler[T]]],
                      run_modes: Optional[Union[str, Iterable[str]]] = None,
                      epoch: Optional[int] = None,
                      ds_id: Optional[Union[str, Iterable[str]]] = None) -> List[T]:
    ...


@overload
def get_current_items(items: Iterable[Union[T, T2, Scheduler[T], Scheduler[T2]]],
                      run_modes: Optional[Union[str, Iterable[str]]] = None,
                      epoch: Optional[int] = None,
                      ds_id: Optional[Union[str, Iterable[str]]] = None) -> List[Union[T, T2]]:
    ...


def get_current_items(items: Iterable[Union[Any, Scheduler[Any]]],
                      run_modes: Optional[Union[str, Iterable[str]]] = None,
                      epoch: Optional[int] = None,
                      ds_id: Optional[Union[str, Iterable[str]]] = None) -> List[Any]:
    """Select items which should be executed for given mode, epoch, and ds_id.

    Args:
        items: A list of possible items or Schedulers of items to choose from.
        run_modes: The desired execution mode. One or more of "train", "eval", "test", or "infer". If None, items of
            all modes will be returned.
        epoch: The desired execution epoch. If None, items across all epochs will be returned.
        ds_id: The desired one or more execution dataset id(s). If None, items across all ds_ids will be returned. An
            empty string indicates that positive matches should be excluded ('' != 'ds1'), but that negative matches are
            satisfied ('' == '!ds1').

    Returns:
        The items which should be executed.
    """
    selected_items = []
    run_modes = to_set(run_modes)
    ds_id = to_set(ds_id)
    for item in items:
        if isinstance(item, Scheduler):
            if epoch is None:
                item = item.get_all_values()
            else:
                item = [item.get_current_value(epoch)]
        else:
            item = [item]
        for item_ in item:
            # mode matching
            mode_match = False
            if not run_modes:
                mode_match = True
            if not hasattr(item_, "mode"):
                mode_match = True
            else:
                if not item_.mode:
                    mode_match = True
                elif item_.mode.intersection(run_modes):
                    mode_match = True

            # ds_id matching
            ds_id_match = False
            if not ds_id:
                ds_id_match = True
            if not hasattr(item_, "ds_id"):
                ds_id_match = True
            else:
                # If the object has no requirements, then allow it
                if not item_.ds_id:
                    ds_id_match = True
                # blacklist check (before whitelist due to desired empty string behavior)
                # if any of ds_id starts with "!", then they will all start with "!"
                elif any([x.startswith("!") for x in item_.ds_id]) and all([x[1:] not in ds_id for x in item_.ds_id]):
                    ds_id_match = True  # Note that empty string will pass this check (unless target is literally "!")
                # whitelist check
                elif item_.ds_id.intersection(ds_id):
                    ds_id_match = True  # Note that empty string will fail this check
            if item_ and mode_match and ds_id_match:
                selected_items.append(item_)
    return selected_items
