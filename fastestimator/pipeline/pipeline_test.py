import tensorflow as tf
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape
import shutil
import numpy as np
import os




class TestPipeline:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    pipeline = Pipeline(feature_name=["x", "y"], train_data={'x': x_train, 'y': y_train},
                            validation_data={"x": x_test, "y": y_test}, batch_size=32,
                        transform_train=[[Reshape([28, 28, 1]), Minmax()],
                                        [Onehot(10)]])
    data = {'x': x_train[0, :], 'y': y_train[0]}
    pipeline._prepare()

    def test_show_batches(self):

        batch = self.pipeline.show_batches(mode='eval', inputs=self.pipeline.inputs, num_batches=2)
        x = np.asarray(batch['x'])
        y = np.asarray(batch['y'])
        assert (x.shape == (2, 32, 28, 28, 1) and y.shape == (2, 32, 10))

    def test_prepare(self):
        assert self.pipeline.feature_name == ['x', 'y']
        assert len(os.listdir(self.pipeline.inputs)) > 0
        assert self.pipeline.validation_data['x'].shape == (10000, 28, 28)
        assert self.pipeline.validation_data['y'].shape == (10000,)

    def test_get_tfrecord_config(self):
        self.pipeline._get_tfrecord_config(self.pipeline.inputs)
        assert self.pipeline.num_examples == {'train': 60000, 'eval': 10000}
        assert self.pipeline.decode_type == {'x': 'uint8', 'y': 'uint8'}


    def test_input_source(self):
        dataset = self.pipeline._input_source(mode='train')
        assert dataset.output_shapes['x'].as_list() == [None, 28, 28, 1]
        assert dataset.output_shapes['y'].as_list() == [None, 10]
        assert dataset.output_types['x'] == tf.float32
        assert dataset.output_types['y'] == tf.float32

    def test_preprocess_fn(self):
        data1 = self.pipeline._preprocess_fn(self.data, mode="train")
        assert not np.all(data1['x'] == self.data['x'])
        assert not np.all(data1['y'] == self.data['y'])


    def test_final_transform(self):
        data1 = self.pipeline.final_transform(self.data)
        assert (data1 == self.data)

    def test_edit_feature(self):
        data1 = self.pipeline.edit_feature(self.data)
        assert data1 == self.data

    def test_split_in_out(self):
        data1 = self.pipeline._split_in_out(self.data)
        assert data1 == self.data

    def test_read_and_decode(self):
        filenames = self.pipeline.file_names["train"]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda dataset: self.pipeline.read_and_decode(dataset))
        assert dataset.output_types['x'] == tf.int32
        assert dataset.output_types['y'] == tf.int32
        shutil.rmtree(self.pipeline.inputs)


