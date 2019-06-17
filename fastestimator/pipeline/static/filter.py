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

    def predicate_fn(self, dataset):
        """
        Filters the dataset based on the filter probabilities.
        
        Args:
            dataset: Tensorflow dataset object which is to be filtered

        Returns:
            Tensorflow conditional for filtering the dataset based on the probabilities for each of the values.
        """
        num_filter = len(self.feature_name)
        for i in range(num_filter):
            keep_prob_i = tf.cast(self.keep_prob[i], tf.float32)
            if i == 0:
                predicate = tf.cond(tf.equal(tf.reshape(dataset[self.feature_name[i]], []), self.filter_value[i]),
                                    lambda: tf.greater(keep_prob_i, tf.random_uniform([])),
                                    lambda: tf.constant(True))
            else:
                predicate = tf.math.logical_and(predicate,
                                                tf.cond(tf.equal(tf.reshape(dataset[self.feature_name[i]], []), self.filter_value[i]),
                                                        lambda: tf.greater(keep_prob_i, tf.random_uniform([])),
                                                        lambda: tf.constant(True)))
        return predicate
