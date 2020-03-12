import pdb
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from fastestimator.op import NumpyOp


class Reshape(NumpyOp):
    """Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]],
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shape = shape
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_reshape(elem) for elem in data]

    def _apply_reshape(self, data):
        data = np.reshape(data, self.shape)
        return data
