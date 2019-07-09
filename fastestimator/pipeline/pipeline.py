from fastestimator.pipeline.static.augmentation import AbstractAugmentation
from fastestimator.util.tfrecord import TFRecorder, get_features
from fastestimator.pipeline.static.filter import Filter
from fastestimator.util.util import convert_tf_dtype
import tensorflow as tf
import multiprocessing
import numpy as np
import time
import json
import os
# import pdb

class Pipeline:
    """
    Class representing the data pipeline required for fastestimator

    Args:
        batch_size: Integer representing the batch size for training model
        feature_name: List of strings representing the feature names in the data (headers in csv, keys in dictionary
            or features in TFRecords)
        transform_train: List of lists of tensor transformations to be performed sequentially on the corresponding
            features.
        transform_dataset: List of lists of numpy transformations to be performed sequentially  on the raw data
            before the TFRecords are made.
        train_data: Training dataset in the form of dictionary containing numpy data, or csv file (with file
            paths or data)
        validation_data: Validation data in the form of dictionary containing numpy data, or csv file, or fraction
            of training data to be sequestered for validation during training
        data_filter: Filtering to be performed on the corresponding features in the form of an object from the
            Filter class
        padded_batch: Whether to pad the batch data in case of inconsistent shape within batch. fill value is 0.
        **kwargs: Additional arguments to be forwarded for the creation of TFRecords.
    """
    def __init__(self,
                 batch_size,
                 feature_name,
                 transform_train,
                 transform_dataset=None,
                 train_data=None,
                 validation_data=None,
                 data_filter=None,
                 padded_batch=False,
                 **kwargs):
        self.batch_size = batch_size
        self.train_data = train_data
        self.feature_name = feature_name
        self.transform_train = transform_train
        self.transform_dataset = transform_dataset
        self.validation_data = validation_data
        self.data_filter = data_filter
        self.padded_batch = padded_batch
        self.kwargs = kwargs
        self.num_process = 1 #change later by mpi
        self.num_local_process = 1 #change later by mpi
        self.rank = 0 #change later by mpi
        self.local_rank = 0 #change later by mpi
        self.decode_type = None #change later by tfrecord config
        self.feature_shape = None #change later by tfrecord config
        self.compression = None

    def _prepare(self, inputs=None):
        """
        Prepares raw data and converts to TFRecords

        Args:
            inputs: Input directory where TFRecords exist

        Returns:

        """
        self.inputs = inputs
        self.num_subprocess = min(8, multiprocessing.cpu_count()//self.num_local_process)
        if self.train_data:
            tfrecorder = TFRecorder(train_data=self.train_data,
                                    feature_name=self.feature_name, 
                                    transform_dataset=self.transform_dataset, 
                                    validation_data=self.validation_data,
                                    **self.kwargs)
            tfrecorder.rank = self.rank
            tfrecorder.local_rank = self.local_rank
            tfrecorder.num_process = self.num_process
            tfrecorder.num_subprocess = self.num_subprocess
            tfrecorder.edit_feature = self.edit_feature
            self.inputs = tfrecorder.create_tfrecord(inputs)
        if self.num_process > 1:
            import horovod.tensorflow.keras as hvd
            hvd.allreduce([0], name="Barrier")
        self._get_tfrecord_config(self.inputs)

    def _get_tfrecord_config(self, data_dir):
        """
        Read TFRecords using summary files

        Args:
            data_dir: Input directory where TFRecords exist

        Returns:

        """
        json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        assert len(json_files) > 0, "Cannot find .json file in %s" % data_dir
        self.file_names = {"train": [], "eval": []}
        self.num_examples = {"train": 0, "eval": 0}
        for json_file in json_files:
            summary = json.load(open(json_file, 'r'))
            if self.decode_type is None:
                self.decode_type = {name:dtype for (name, dtype) in zip(summary["feature_name"], summary["feature_dtype"])}
            if self.feature_shape is None:
                self.feature_shape = {name:dtype for (name, dtype) in zip(summary["feature_name"], summary["feature_shape"])}
                if "compression" in summary:
                    self.compression = summary["compression"]
            if "eval_files" in summary:
                self.file_names["eval"].extend([os.path.join(data_dir, f) for f in summary["eval_files"]])
                self.num_examples["eval"] += np.sum(summary["num_eval_examples"])
            self.file_names["train"].extend([os.path.join(data_dir, f) for f in summary["train_files"]])
            self.num_examples["train"] += np.sum(summary["num_train_examples"])
        self.keys_to_features = get_features(self.file_names["train"][0], compression=self.compression)
        assert len(self.file_names["train"]) >= self.num_local_process, "number of training file per local process should at least be 1"
        if self.local_rank == 0:
            print("FastEstimator: Found %d examples for training and %d for validation in %s" %(self.num_examples["train"], self.num_examples["eval"], data_dir))

    def _input_stream(self, mode):
        """
        Prepares data from TFRecords for streaming input

        Args:
            mode: Mode for current pipeline ("train", "eval" or "both")

        Returns:
            Dataset object containing the batch of tensors to be ingested by the model
        """
        filenames = self.file_names[mode]
        # files reading
        if mode == "train":
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.shard(self.num_local_process, self.local_rank)
            dataset = dataset.shuffle(len(filenames))
            dataset = dataset.interleave(lambda dataset: tf.data.TFRecordDataset(dataset, compression_type=self.compression), cycle_length=self.num_subprocess, block_length=2)
            dataset = dataset.shuffle(min(10000, self.num_examples[mode]))
            dataset = dataset.repeat()
        else:
            dataset = tf.data.TFRecordDataset(filenames, compression_type=self.compression)
        # reading and decoding
        dataset = dataset.map(lambda dataset: self.read_and_decode(dataset), num_parallel_calls=self.num_subprocess)
        # filtering and preprocessing
        if isinstance(self.data_filter, list):
            assert len(self.data_filter) > 1, "must provide at least two data filters for dataset zipping"
            zip_ds = ()
            for data_filter in self.data_filter:
                assert isinstance(data_filter, Filter), "must provide Filter instance"
                ds = dataset.filter(lambda dataset: data_filter.filter_fn(dataset))
                ds = ds.map(lambda ds: self._preprocess_fn(ds, mode), num_parallel_calls=self.num_subprocess)
                zip_ds += ds,
            dataset = tf.data.Dataset.zip(zip_ds)
        else:
            if isinstance(self.data_filter, Filter) and self.data_filter.mode in [mode, "both"]:
                dataset = dataset.filter(lambda dataset: self.data_filter.filter_fn(dataset))
            dataset = dataset.map(lambda dataset: self._preprocess_fn(dataset, mode), num_parallel_calls=self.num_subprocess)
        #batching
        if self.padded_batch:
            dataset = dataset.padded_batch(self.batch_size, padded_shapes={key:self.feature_shape[key] for key in self.feature_name})
        else:
            dataset = dataset.batch(self.batch_size)
        #prefetching
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _preprocess_fn(self, decoded_data, mode):
        """
        Preprocessing performed on the tensor data in features in the order specified in the transform_train list
        Args:
            decoded_data: dataset object containing a dictionary of tensors
            mode: Mode for training ("train", "eval" or "both")
        Returns:
            Dictionary containing the preprocessed data in the form of a dictionary of tensors
        """
        preprocessed_data = {}
        randomized_list = []
        for idx in range(len(self.feature_name)):
            transform_list = self.transform_train[idx]
            feature_name = self.feature_name[idx]
            preprocess_data = decoded_data[feature_name]
            for preprocess_obj in transform_list:
                preprocess_obj.feature_name = feature_name
                preprocess_obj.decoded_data = decoded_data
                if isinstance(preprocess_obj, AbstractAugmentation):
                    if preprocess_obj.mode == mode or preprocess_obj.mode == "both":
                        preprocess_obj.height = preprocess_data.get_shape()[0]
                        preprocess_obj.width = preprocess_data.get_shape()[1]
                        if preprocess_obj not in randomized_list:
                            preprocess_obj.setup()
                            randomized_list.append(preprocess_obj)
                preprocess_data = preprocess_obj.transform(preprocess_data)
            preprocessed_data[feature_name] = preprocess_data
        return preprocessed_data

    def _input_source(self, mode, num_steps):
        """Package the data from tfrecord to model
        
        Args:
            mode (str): mode for current pipeline ("train", "eval" or "both")
        
        Returns:
            Iterator: An iterator that can provide a streaming of processed data
        """
        dataset = self._input_stream(mode)
        for batch_data in dataset.take(num_steps):
            batch_data = self.final_transform(batch_data)
            yield batch_data

    def final_transform(self, preprocessed_data):
        """
        Can be overloaded to change tensors in any manner

        Args:
            preprocessed_data: Batch of training data as a tf.data object

        Returns:
            A dictionary of tensor data in the form of a tf.data object.
        """
        return preprocessed_data

    def edit_feature(self, feature):
        """
        Can be overloaded to change raw data dictionary in any manner

        Args:
            feature: Dictionary containing the raw data

        Returns:
            Dictionary containing raw data to be stored in TFRecords

        """
        return feature

    def read_and_decode(self, dataset):
        """
        Reads and decodes the string data from TFRecords

        Args:
            dataset: Dataset consisting of encoded data from TFRecords

        Returns:
            Dictionary of decoded data

        """
        decoded_data = {}
        all_data = tf.io.parse_single_example(dataset, features=self.keys_to_features)
        for feature in self.feature_name:
            data = all_data[feature]
            if "string" in str(data.dtype) and "string" not in self.decode_type[feature]:
                data = tf.io.decode_raw(data, convert_tf_dtype(self.decode_type[feature]))
                data = tf.reshape(data, self.feature_shape[feature])
            if "int" in str(data.dtype):
                data = tf.cast(data, tf.int32)
            elif self.decode_type[feature] == "string":
                data = data
            else:
                data = tf.cast(data, tf.float32)
            decoded_data[feature] = data
        return decoded_data

    def show_batches(self, inputs=None, num_batches=1, mode="train"):
        """
        Shows batches of tensor data in numpy

        Args:
            inputs: Directory for saving TFRecords
            num_batches: Number of batches to show
            mode: Mode for training ("train", "eval" or "both")

        Returns:
            A dictionary containing the batches data
        """
        data = []
        self.num_subprocess = min(8, multiprocessing.cpu_count()//self.num_local_process)
        if self.train_data:
            tfrecorder = TFRecorder(train_data=self.train_data,
                                    feature_name=self.feature_name, 
                                    transform_dataset=self.transform_dataset, 
                                    validation_data=self.validation_data,
                                    **self.kwargs)
            tfrecorder.edit_feature = self.edit_feature
            tfrecorder.create_tfrecord(inputs)
            inputs = tfrecorder.save_dir
        else:
            assert inputs is not None, "Must specify the data path when using existing tfrecords"
        self._get_tfrecord_config(inputs)
        dataset = self._input_source(mode, num_batches)
        for i, example in enumerate(dataset):
            data.append(example)
        return data

    def benchmark(self, inputs=None, mode="train", num_steps= 500, log_interval= 100):
        """
        benchmark the pipeline processing speed during training

        Args:
            inputs: Directory for saving TFRecords
            mode: Mode for training ("train", "eval" or "both")
        """
        self.num_subprocess = min(8, multiprocessing.cpu_count()//self.num_local_process)
        if self.train_data:
            tfrecorder = TFRecorder(train_data=self.train_data,
                                    feature_name=self.feature_name, 
                                    transform_dataset=self.transform_dataset, 
                                    validation_data=self.validation_data,
                                    **self.kwargs)
            tfrecorder.edit_feature = self.edit_feature
            tfrecorder.create_tfrecord(inputs)
            inputs = tfrecorder.save_dir
        else:
            assert inputs is not None, "Must specify the data path when using existing tfrecords"
        self._get_tfrecord_config(inputs)
        it = self._input_source(mode, num_steps)
        start = time.time()
        for i, _ in enumerate(it):
            if i % log_interval == 0 and i >0:
                duration = time.time() - start
                example_per_sec = log_interval * self.batch_size / duration
                print("FastEstimator: Pipeline Preprocessing Example/sec %f" % example_per_sec)
                start = time.time()
