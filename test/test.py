from fastestimator.pipeline.dynamic.preprocess import ImageReader
from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape, AbstractPreprocessing
from fastestimator.pipeline.static.augmentation import AbstractAugmentation
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.network import Network
import tensorflow as tf
from fastestimator.application.lenet import LeNet as lenet
import tensorflow.keras.backend as K
from fastestimator.estimator.callbacks import LearningRateScheduler, ReduceLROnPlateau
from fastestimator.pipeline.static.filter import Filter
from fastestimator.network.lrscheduler import CyclicScheduler
import numpy as np


class custom_pipeline(Pipeline):
    def edit_feature(self, feature):
        return {'x': feature['x'] * 2, 'y': feature['y']}

    def read_and_decode(self, dataset):
        keys_to_data = {"x": tf.FixedLenFeature([], tf.string), "y": tf.FixedLenFeature([], tf.int64)}
        all_data = tf.parse_single_example(dataset, features = keys_to_data)
        x = all_data["x"]
        x = tf.decode_raw(x, "uint8")
        x = tf.cast(x, tf.float32)
        y = all_data["y"]
        y = tf.cast(y, tf.int32)
        decoded_data = {"x": x, "y": y}
        return decoded_data

    def final_transform(self, preprocessed_data):
        return {'x': tf.add(preprocessed_data['x'], 2),
                'y': preprocessed_data['y']}


class customized_normalization(AbstractPreprocessing):
    def transform(self, data, decoded_data=None):
        data = tf.cast(data, tf.float32)
        data = tf.div(data, tf.maximum(
                tf.subtract(tf.reduce_max(data), tf.reduce_min(data)), 1e-4))
        return data


class customized_zoom(AbstractAugmentation):
    def __init__(self, zoom_range, mode="train"):
        self.zoom_range = zoom_range
        self.mode = mode

    def setup(self):
        pass

    def transform_matrix_offset_center(self, matrix):
        """
        :param matrix: Affine tensor
        :return: An affine tensor offset to the center of the image
        """
        o_x = self.height / tf.constant([2], dtype=tf.float32) + tf.constant([0.5], dtype=tf.float32)
        o_y = self.width / tf.constant([2], dtype=tf.float32) + tf.constant([0.5], dtype=tf.float32)
        eye = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32)

        offset_matrix = eye + \
                        tf.multiply(o_x,
                                    tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)) + \
                        tf.multiply(o_y, tf.constant([[0, 0, 0], [0, 0, 1], [0, 0, 0]], shape=[3, 3], dtype=tf.float32))

        reset_matrix = eye + \
                       tf.multiply(o_x,
                                   tf.constant([[0, 0, -1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)) + \
                       tf.multiply(tf.constant([[0, 0, 0], [0, 0, -1], [0, 0, 0]], shape=[3, 3], dtype=tf.float32), o_y)

        transform_matrix = tf.tensordot(tf.tensordot(offset_matrix, matrix, axes=1), reset_matrix, axes=1)
        return transform_matrix

    def transform(self, data):
        """
        :return: An affine transformation tensor for a random 2D zooming / scaling
        """
        zoom_range = [0., 0.]
        if type(self.zoom_range) is not tuple and type(self.zoom_range) is not list:
            zoom_range[0] = 1 - self.zoom_range
            zoom_range[1] = 1 + self.zoom_range
        else:
            zoom_range = self.zoom_range
        self.zoom_range = zoom_range
        zx = tf.random_uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        zy = tf.random_uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        base_zx = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zx
        base_zy = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zy
        transform_matrix = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32) + base_zx + base_zy
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        transform_matrix_flatten = tf.reshape(transform_matrix, shape=[1, 9])
        transform_matrix_flatten = transform_matrix_flatten[0, 0:8]
        augment_data = tf.contrib.image.transform(data, transform_matrix_flatten)
        return augment_data


def rmse_loss(y_true, y_pred):
    rmse = K.sqrt(K.mean(K.square(y_true - y_pred)))
    return rmse

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

class MnistFilter(Filter):
    def __init__(self, mode):
        self.mode = mode

    def predicate_fn(self, dataset):
        #if the label is 5, then there's 50% chance to drop the example
        predicate = tf.cond(tf.equal(tf.reshape(dataset["y"], []), 5),
                            lambda: tf.greater(0.5, tf.random_uniform([])),
                            lambda: tf.constant(True))
        return predicate




class my_lr_schedule(CyclicScheduler):
    def __init__(self):
        self.mode = "global_steps"

    def lr_schedule_fn(self, global_steps):
        lr_ratio = 1e-6
        if global_steps <= 500:
            lr_ratio = 1.0
        elif global_steps <= 1500:
            lr_ratio = self.lr_linear_decay(global_steps, lr_ratio_start=1.0, lr_ratio_end=0.5, step_start=500, step_end=1500)
        elif global_steps <= 3000:
            lr_ratio = self.lr_cosine_decay(global_steps, lr_ratio_start=0.5, lr_ratio_end=1e-6, step_start=1500, step_end=3000)
        return np.float32(lr_ratio)


def get_estimator(epochs=2, batch_size=32, data_type="numpy",
                  train_csv='annotation_train.csv', val_split=None,
                  val_csv='annotation_val.csv', loss=None, metrics=None,
                  loss_weights=None, optimizer=None, model_dir=None,
                  filter=None, steps_per_epoch=10, validation_steps=10,
                  lr_schedule=None, decrease_method="cosine", num_cycle=2,
                  pipeline=None, preprocess=None, augment=None, reduce_lr=None, compression=None):

    # Define parameters
    callbacks = []
    pipeline_filter = None
    if filter == 'regular':
        pipeline_filter = Filter('y', [0, 1, 5], [0.6, 0.8, 0.1])
    elif filter == 'custom':
        pipeline_filter = MnistFilter(mode='train')

    network_loss = 'categorical_crossentropy'
    if loss == 'custom':
        network_loss = rmse_loss

    network_metrics = None
    if metrics == 'custom':
        network_metrics = [mean_pred]
    elif metrics == 'regular':
        network_metrics = ['acc']

    network_loss_weights = {'y': 0.9}
    if loss_weights == "None":
        network_loss_weights = None

    network_optimizer = "adam"
    if optimizer == 'custom':
        network_optimizer = tf.keras.optimizers.Adam(lr=0.001)

    network_model_dir = None
    if model_dir is not None:
        network_model_dir = model_dir

    if reduce_lr is not None:
        callbacks.append(ReduceLROnPlateau())

    if lr_schedule == 'regular':
        callbacks.append(
            LearningRateScheduler(schedule=CyclicScheduler(decrease_method=decrease_method, num_cycle=num_cycle)))
    elif lr_schedule == 'custom':
        callbacks.append(
            LearningRateScheduler(schedule=my_lr_schedule()))

    my_pipeline = Pipeline
    if pipeline == 'custom':
        my_pipeline = custom_pipeline

    data_transform = [[Reshape([28, 28, 1]), Minmax()], [Onehot(10)]]

    if preprocess == 'custom':
        data_transform[0].append(customized_normalization())

    if augment == 'custom':
        data_transform[0].append(customized_zoom(zoom_range=[0.9, 1.1]))

    # Step 1: Define Pipeline

    pipeline = None
    if data_type == "csv":
        if val_split is not None:
            val = val_split
        else:
            val = val_csv
        pipeline = my_pipeline(batch_size=batch_size,
                            feature_name=["x", "y"],
                            train_data=train_csv,
                            validation_data=val,
                            data_filter=pipeline_filter,
                            transform_dataset=[[ImageReader(grey_scale=True)], []],
                            transform_train=data_transform,
                            compression=compression)
    elif data_type == 'numpy':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if val_split is not None:
            val = val_split
        else:
            val = {'x': x_test, 'y': y_test}
        pipeline = my_pipeline(batch_size=batch_size,
                            feature_name=["x", "y"],
                            train_data={'x': x_train, 'y': y_train},
                            data_filter=pipeline_filter,
                            validation_data=val,
                            transform_train=data_transform,
                            compression=compression)
    elif data_type == 'tfrecords':
        pipeline = my_pipeline(batch_size=batch_size,
                            feature_name=["x", "y"],
                            data_filter=pipeline_filter,
                            transform_train=data_transform,
                            compression=compression)

    # Step2: Define Network

    network = Network(model=lenet('x', 'y'),
                      loss=network_loss,
                      metrics=network_metrics,
                      loss_weights=network_loss_weights,
                      optimizer=network_optimizer,
                      model_dir=network_model_dir)




    # Step3: Define Estimator and fit
    estimator = Estimator(network=network,
                          pipeline=pipeline,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          epochs=epochs,
                          callbacks=callbacks)
    return estimator
