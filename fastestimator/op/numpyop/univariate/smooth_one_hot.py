from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np

from fastestimator.op import NumpyOp


class SmoothOneHot(NumpyOp):
    """ Transform the non-one-hot encoding label to one-hot-encoding and smooth the label
        Ref: https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

    Args:
        class_num: Total class number.
        label_smoothing: Smoothing factor. The class index value will become: 1 - label_smoothing + label_smoothing
            / class_num and The other class index value will become 1 / class_num
        inputs: Input key(s) of labels to be smoothed
        outputs: Output key(s) of labels
        mode: What execution mode (train, eval, None) to apply this operation
    """
    def __init__(self,
                 class_num: int,
                 label_smoothing: float,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.class_num = class_num
        self.label_smoothing = label_smoothing

    def forward(self, data: Union[int, np.ndarray],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:

        if isinstance(data, int):
            class_index = data
        elif isinstance(data, np.ndarray):
            assert len(data.shape) == 1 and data.shape[0] == 1, "the data array should have only 1 dimension, and one element"
            class_index = data[0]
        else:
            assert False, "the data should be int or ndarray"

        assert class_index < self.class_num, "label value should be smaller than class_num"

        output = np.full((self.class_num), fill_value=self.label_smoothing / self.class_num)
        output[class_index] = 1 - self.label_smoothing + self.label_smoothing / self.class_num

        return output
