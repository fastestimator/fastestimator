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
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union

if TYPE_CHECKING:
    # Hide these imports for speed
    import numpy as np
    import tensorflow as tf
    import torch

    T = TypeVar('T')

    Tensor = Union[torch.Tensor, tf.Tensor]
    Array = Union[np.ndarray, Tensor]
    DataSequence = Union[Sequence, Array]
else:
    # The following allow runtime isinstance() checks against the types defined above, without incurring import speed
    # penalties if the user doesn't run the isinstance check (by hiding the tf import)

    class _MetaTensor(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import tensorflow as tf
            import torch
            return isinstance(__instance, (tf.Tensor, torch.Tensor))

        def __subclasscheck__(self, __subclass: type) -> bool:
            import tensorflow as tf
            import torch
            return issubclass(__subclass, (tf.Tensor, torch.Tensor))

    class Tensor(metaclass=_MetaTensor):
        ...

    class _MetaArray(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import numpy as np
            import tensorflow as tf
            import torch
            return isinstance(__instance, (tf.Tensor, torch.Tensor, np.ndarray))

        def __subclasscheck__(self, __subclass: type) -> bool:
            import numpy as np
            import tensorflow as tf
            import torch
            return issubclass(__subclass, (tf.Tensor, torch.Tensor, np.ndarray))

    class Array(metaclass=_MetaArray):
        ...

    class _MetaDataSequence(type):
        def __instancecheck__(self, __instance: Any) -> bool:
            import numpy as np
            import tensorflow as tf
            import torch
            return isinstance(__instance, (Sequence, tf.Tensor, torch.Tensor, np.ndarray))

        def __subclasscheck__(self, __subclass: type) -> bool:
            import numpy as np
            import tensorflow as tf
            import torch
            return issubclass(__subclass, (Sequence, tf.Tensor, torch.Tensor, np.ndarray))

    class DataSequence(metaclass=_MetaDataSequence):
        ...


if __name__ == "__main__":
    import torch
    print(isinstance({'a': 5}, DataSequence))
    print(isinstance(torch.ones((5)), DataSequence))
