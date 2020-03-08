from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from fastestimator.op import NumpyOp


class Onehot(NumpyOp):
    """ Transform the label integer to one-hot-encoding
        Ref: https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

    Args:
        num_classes: Total number of classes.
        label_smoothing: Smoothing factor. The class index value will become: 1 - label_smoothing + label_smoothing
            / num_classes and The other class index value will become label_smoothing / num_classes
        inputs: Input key(s) of labels to be onehot encoded
        outputs: Output key(s) of labels
        mode: What execution mode (train, eval, None) to apply this operation
    """
    def __init__(self,
                 num_classes: int,
                 label_smoothing: Optional[float] = None,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        if not self.label_smoothing:
            self.label_smoothing = 0.0

    def forward(self, data: Union[int, np.ndarray], state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        class_index = np.array(data)
        assert "int" in str(class_index.dtype)
        assert class_index.size == 1, "data must have only one item"
        class_index = class_index.item()
        assert class_index < self.num_classes, "label value should be smaller than num_classes"
        output = np.full((self.num_classes), fill_value=self.label_smoothing / self.num_classes)
        output[class_index] = 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes
        return output
