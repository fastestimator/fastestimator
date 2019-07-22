import tensorflow as tf

epsilon = 1e-7


class AbstractPreprocessing():
    """
    An abstract class for preprocessing
    """
    def __init__(self):
        self.feature_name = None
        self.decoded_data = None

    def transform(self, data):
        """
        Placeholder function that is to be inherited by preprocessing classes.

        Args:
            data: Data to be preprocessed

        Returns:
            Transformed data tensor
        """
        return data


class Binarize(AbstractPreprocessing):
    """
    Binarize data based on threshold between 0 and 1

    Args:
        threshold: Threshold for binarizing
    """
    def __init__(self, threshold):
        self.thresh = threshold

    def transform(self, data, ):
        """
        Transforms the image to binary based on threshold

        Args:
            data: Data to be binarized

        Returns:
            Tensor containing binarized data
        """
        data = tf.math.greater(data, self.thresh)
        data = tf.cast(data, tf.float32)
        return data


class Zscore(AbstractPreprocessing):
    """
    Standardize data using zscore method
    """
    def transform(self, data):
        """
        Standardizes the data tensor

        Args:
            data: Data to be standardized

        Returns:
            Tensor containing standardized data
        """
        data = tf.cast(data, tf.float32)
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(
            tf.subtract(data, mean),
            tf.maximum(std, epsilon))
        data = tf.cast(data, tf.float32)
        return data


class Minmax(AbstractPreprocessing):
    """
    Normalize data using the minmax method
    """
    def transform(self, data):
        """
        Normalizes the data tensor

        Args:
            data: Data to be normalized

        Returns:
            Tensor after minmax
        """
        data = tf.cast(data, tf.float32)
        data = tf.math.divide(
            tf.subtract(data, tf.reduce_min(data)),
            tf.maximum(
                tf.subtract(tf.reduce_max(data), tf.reduce_min(data)), epsilon))
        return data


class Scale(AbstractPreprocessing):
    """
    Preprocessing class for scaling dataset

    Args:
        scalar: Scalar for scaling the data
    """
    def __init__(self, scalar):
        self.scalar = scalar

    def transform(self, data):
        """
        Scales the data tensor

        Args:
            data: Data to be scaled

        Returns:
            Scaled data tensor
        """
        data = tf.cast(data, tf.float32)
        data = tf.scalar_mul(self.scalar, data)
        return data


class Onehot(AbstractPreprocessing):
    """
    Preprocessing class for converting categorical labels to onehot encoding

    Args:
        num_dim: Number of dimensions of the labels
    """
    def __init__(self, num_dim):
        self.num_dim = num_dim

    def transform(self, data):
        """
        Transforms categorical labels to onehot encodings

        Args:
            data: Data to be preprocessed

        Returns:
            Transformed labels
        """
        data = tf.cast(data, tf.int32)
        data = tf.one_hot(data, self.num_dim)
        return data

    
class Resize(AbstractPreprocessing):
    """
    Preprocessing class for resizing the images

    Args:
        size: Destination shape of the images
        resize_method: One of resize methods provided by tensorflow to be used
    """
    def __init__(self, size, resize_method=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.resize_method = resize_method

    def transform(self, data):
        """
        Resizes data tensor

        Args:
            data: Tensor to be resized

        Returns:
            Resized tensor
        """
        preprocessed_data = tf.image.resize_images(data, self.size, self.resize_method)
        return preprocessed_data


class Reshape(AbstractPreprocessing):
    """
    Preprocessing class for reshaping the data

    Args:
        shape: target shape
    """
    def __init__(self, shape):
        self.shape = shape

    def transform(self, data):
        """
        Reshapes data tensor
        
        Args:
            data: Data to be reshaped

        Returns:
            Reshaped tensor
        """
        data = tf.reshape(data, self.shape)
        return data
