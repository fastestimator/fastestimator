import tensorflow as tf

from fastestimator.op.tensorop.loss import Loss


class SparseCategoricalCrossentropy(Loss):
    """Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed to
    tf.losses.SparseCategoricalCrossentropy

    Args:
        y_true: ground truth label key
        y_pred: prediction label key
        inputs: A tuple or list like: [<y_true>, <y_pred>]
        outputs: Where to store the computed loss value (not required under normal use cases)
        mode: 'train', 'eval', 'test', or None
        kwargs: Arguments to be passed along to the tf.losses constructor. Passing the 'reduction' arg will raise a
                KeyError
    """
    def __init__(self, y_true=None, y_pred=None, inputs=None, outputs=None, mode=None, **kwargs):
        if 'reduction' in kwargs:
            raise KeyError("parameter 'reduction' not allowed")
        inputs = self.validate_loss_inputs(inputs, y_true, y_pred)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = tf.losses.SparseCategoricalCrossentropy(reduction='none', **kwargs)

    def forward(self, data, state):
        y_true, y_pred = data
        return self.loss_obj(y_true, y_pred)
