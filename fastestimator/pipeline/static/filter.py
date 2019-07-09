import tensorflow as tf


class Filter:
    """
    Class for performing filtering on dataset based on scalar values.

    Args:
        feature_name: Name of the key in the dataset that is to be filtered
        filter_value: The values in the dataset that are to be filtered.
        keep_prob: The probability of keeping the example
        mode: filter on 'train', 'eval' or 'both'
    """
    def __init__(self, feature_name, filter_value, keep_prob, mode="train"):
        self.feature_name = feature_name
        self.filter_value = filter_value
        self.keep_prob = keep_prob
        self.mode = mode

    def filter_fn(self, dataset):
        return tf.cond(tf.equal(tf.reshape(dataset[self.feature_name], []), self.filter_value),
                                    lambda: tf.greater(self.keep_prob, tf.random.uniform([])),
                                    lambda: tf.constant(True))