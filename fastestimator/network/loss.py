import tensorflow as tf

class Loss:
    def __init__(self):
        self.loss_obj = None
        self.loss_fn = None

    def calculate_loss(self, batch):
        """this is the function that calculates the loss given the batch data
        
        Args:
            batch (dict): batch data after forward operations
        
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

    def calculate_loss(self, batch):
        loss = self.loss_obj(batch[self.true_key], batch[self.pred_key])
        return loss


