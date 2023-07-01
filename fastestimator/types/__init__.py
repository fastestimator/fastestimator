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
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, List, Optional, Protocol, Sequence, Sized, TypeVar, \
    Union, runtime_checkable


class FilteredData:
    """A placeholder to indicate that this data instance should not be used.

    This class is intentionally not @traceable.

    Args:
        replacement: Whether to replace the filtered element with another (thus maintaining the number of steps in an
            epoch but potentially increasing data repetition) or else shortening the epoch by the number of filtered
            data points (fewer steps per epoch than expected, but no extra data repetition). Either way, the number of
            data points within an individual batch will remain the same. Even if `replacement` is true, data will not be
            repeated until all of the given epoch's data has been traversed (except for at most 1 batch of data which
            might not appear until after the re-shuffle has occurred).
    """
    def __init__(self, replacement: bool = True):
        self.replacement = replacement

    def __repr__(self):
        return "FilteredData"


@runtime_checkable
class MapDataset(Sized, Protocol):
    def __getitem__(self, index: int) -> Union[Dict[str, Any], List[Dict[str, Any]], FilteredData]:
        ...

    fe_batch: Optional[int]
    fe_reset_ds: Optional[Callable[[bool], None]]
    fe_batch_indices: Optional[Callable[[int], List[List[int]]]]


CollectionT = TypeVar('CollectionT', bound=Collection)

if TYPE_CHECKING:
    # Hide these imports for speed
    import numpy as np
    import tensorflow as tf
    import torch

    Tensor = Union[torch.Tensor, tf.Tensor, tf.Variable]
    Array = Union[np.ndarray, Tensor]
    DataSequence = Union[Sequence, Array]
    Model = Union[tf.keras.Model, torch.nn.Module]

    # Use these when you want to indicate that you will return the same class that was input
    TensorT = TypeVar('TensorT', torch.Tensor, tf.Tensor, tf.Variable)
    ArrayT = TypeVar('ArrayT', torch.Tensor, tf.Tensor, tf.Variable, np.ndarray)
    ModelT = TypeVar('ModelT', tf.keras.Model, torch.nn.Module)

else:
    TensorT = TypeVar('TensorT')
    ArrayT = TypeVar('ArrayT')
    ModelT = TypeVar('ModelT')

    # The following allow runtime isinstance() checks against the types defined above, without incurring import speed
    # penalties if the user doesn't run the isinstance check (by hiding the tf import). In python 3.10+ we could simply
    # do isinstance() checks on Union[] types, but even in that case we would incur speed penalty from importing
    # tensorflow during the Union definition, which would make the types unsuitable for fast-path code like the log
    # visualization CLI.


    class _MetaTensor(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import tensorflow as tf
            import torch
            return isinstance(__instance, torch.Tensor) or tf.is_tensor(__instance)

        def __subclasscheck__(self, __subclass: type) -> bool:
            import tensorflow as tf
            import torch
            return issubclass(__subclass, (tf.Tensor, torch.Tensor))

    class Tensor(metaclass=_MetaTensor):
        ...

    class _MetaArray(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import numpy as np
            return isinstance(__instance, (np.ndarray, Tensor))

        def __subclasscheck__(self, __subclass: type) -> bool:
            import numpy as np
            return issubclass(__subclass, (np.ndarray, Tensor))

    class Array(metaclass=_MetaArray):
        ...

    class _MetaDataSequence(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            return isinstance(__instance, (Sequence, Array))

        def __subclasscheck__(self, __subclass: type) -> bool:

            return issubclass(__subclass, (Sequence, Array))

    class DataSequence(metaclass=_MetaDataSequence):
        ...

    class _MetaModel(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import tensorflow as tf
            import torch
            return isinstance(__instance, (tf.keras.Model, torch.nn.Module))

        def __subclasscheck__(self, __subclass: type) -> bool:
            import tensorflow as tf
            import torch
            return issubclass(__subclass, (tf.keras.Model, torch.nn.Module))

    class Model(metaclass=_MetaModel):
        ...
