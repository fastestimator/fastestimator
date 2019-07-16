from fastestimator.network.loss import Loss
import tensorflow as tf

class Network:
    def __init__(self, ops, model_list=None):
        self.ops = ops
        self._check_op()

    def _check_op(self):
        self.model_list = []
        if not isinstance(self.ops, list):
            self.ops = [self.ops]
        for op in self.ops:
            for block in op.link:
                if isinstance(block, tf.keras.Model) and block not in self.model_list:
                    assert block.fe_compiled is True, "must use prepare_model to compile the keras model before use"
                    self.model_list.append(block)
        assert len(self.model_list) > 0, "network should have at least one model"
        self.num_model = len(self.model_list)

    def forward(self, batch, mode, epoch):
        prediction = {}
        for op in self.ops:
            prediction = op.forward(batch, prediction, mode, epoch)
        return prediction


def prepare_model(keras_model, loss, optimizer):
    assert isinstance(keras_model, tf.keras.Model), "must provide tf.keras.Model instance as keras model"
    assert isinstance(loss, Loss)
    keras_model.loss = loss
    if isinstance(optimizer, str):
        optimizer_fn = {'adadelta': tf.optimizers.Adadelta,
                        'adagrad': tf.optimizers.Adagrad,
                        'adam': tf.optimizers.Adam,
                        'adamax': tf.optimizers.Adamax,
                        'nadam': tf.optimizers.Nadam,
                        'rmsprop': tf.optimizers.RMSprop,
                        'sgd': tf.optimizers.SGD}
        keras_model.optimizer = optimizer_fn[optimizer]
    else:
        assert isinstance(optimizer, tf.optimizers.Optimizer), "must provide tf.optimizer.Optimizer instance as optimizer"
        keras_model.optimizer = optimizer
        keras_model.fe_compiled = True
    return keras_model