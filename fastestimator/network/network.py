from tensorflow.keras import backend as K
import tensorflow as tf
import tempfile
import os

class Network:
    """
    Class for representing the model for fastestimator
    
    Args:
        model: An instance of tensorflow.keras model object.
        loss: String or list or dictionary of strings representing a loss function (defined in keras)
            or it can be a function handle of a customized loss function that takes a true value and
            predicted value and returns a scalar loss value.
        metrics: List or dictionary of strings representing metrics (defined in keras)
            or it can be a list of function handles of a customized metric function that
            takes a true values and predicted values and returns a scalar metric.
        loss_weights: List of floats used only if the loss is a weighted loss
            with the individual components defined as a list in the "loss member variable" (default: ``None``)
        optimizer: the type the optimizer for the model in the form of a string.
        model_dir: Directory where the model is to be saved (default is the temporary directory)
    """
    def __init__(self, model, loss, metrics=None, loss_weights=None, optimizer="adam", model_dir=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self._create_optimizer_map()
        self._check_optimizer()
        if model_dir is None:
            model_dir = tempfile.mkdtemp()
        elif not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir

    def _check_optimizer(self):
        if isinstance(self.optimizer, str):
            self.optimizer = self.optimizer_fn[self.optimizer]()

        if not isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError("optimizer must come from tf.keras.optimizer")

    def _create_optimizer_map(self):
        self.optimizer_fn = {'adadelta': tf.keras.optimizers.Adadelta,
                            'adagrad': tf.keras.optimizers.Adagrad,
                            'adam': tf.keras.optimizers.Adam,
                            'adamax': tf.keras.optimizers.Adamax,
                            'nadam': tf.keras.optimizers.Nadam,
                            'rmsprop': tf.keras.optimizers.RMSprop,
                            'sgd': tf.keras.optimizers.SGD}

    def _get_inout_list(self):
        self.input_names = []
        self.output_names = []
        model_input_list = self.model.input
        if not type(model_input_list) is list:
            model_input_list = [model_input_list]
        for model_input in model_input_list:
            self.input_names.append(self._parse_inout_name(model_input.name.encode("ascii")))
        model_output_list = self.model.output
        if not type(model_output_list) is list:
            model_output_list = [model_output_list]
        for model_output in model_output_list:
            self.output_names.append(self._parse_inout_name(model_output.name.encode("ascii")))

    def _parse_inout_name(self, raw_name):
        if hasattr(raw_name, 'decode'):
            raw_name = raw_name.decode()
        for pos, char in enumerate(raw_name):
            if char in ["/", ":"]:
                break   
        return raw_name[:pos]
