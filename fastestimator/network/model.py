import tensorflow as tf
from fastestimator.network.loss import Loss

def build(keras_model, loss, optimizer):
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
        keras_model.optimizer = optimizer_fn[optimizer]()
    else:
        assert isinstance(optimizer, tf.optimizers.Optimizer), "must provide tf.optimizer.Optimizer instance as optimizer"
        keras_model.optimizer = optimizer
    keras_model.fe_compiled = True
    return keras_model

class ModelOp:
    def __init__(self, model, inputs=None, outputs=None, mode=None):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        assert isinstance(self.model, tf.keras.Model) and self.model.fe_compiled is True, "must prepare your the keras model before use in ModelOp"

    def forward(self, data, mode):
        data = self.model(data, training=mode=="train")
        return data
