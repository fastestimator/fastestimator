import tensorflow as tf

class Loss:
    def __init__(self):
        pass

    def calculate_loss(self, batch, prediction):
        """this is the function that calculates the loss given the batch data
        
        Args:
            batch (dict): batch data before forward operation
            prediction(dict): prediction data after forward operation
        
        Returns:
            loss (scalar): scalar loss for the model update
        """
        loss = None
        return loss


class SparseCategoricalCrossentropy(Loss):
    def __init__(self, true_key, pred_key, **kwargs):
        """Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed to tf.losses.SparseCategoricalCrossentropy
        
        Args:
            true_key (str): the key of ground truth label in batch data
            pred_key (str): the key of predicted label in batch data
        """
        self.true_key = true_key
        self.pred_key = pred_key
        self.loss_obj = tf.losses.SparseCategoricalCrossentropy(**kwargs)

    def calculate_loss(self, batch, prediction):
        loss = self.loss_obj(batch[self.true_key], prediction[self.pred_key])
        return loss


