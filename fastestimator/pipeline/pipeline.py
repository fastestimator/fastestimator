from fastestimator.pipeline.static.augmentation import AbstractAugmentation as StaticAugmentation
from fastestimator.util.tfrecord import TFRecorder, get_features
from fastestimator.util.util import convert_tf_dtype
import tensorflow as tf
import multiprocessing
import numpy as np
import json
import os

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
        data_filter: Filtering to be performed on the corresponding features in the form of an object from the Filter class
        shuffle_buffer: buffer size for the shuffling, it can affect the memory consumption during training. default is 'auto'.
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
                 shuffle_buffer="auto",
                 **kwargs):
        self.batch_size = batch_size
        self.train_data = train_data
        self.feature_name = feature_name
        self.transform_train = transform_train
        self.transform_dataset = transform_dataset
        self.validation_data = validation_data
        self.data_filter = data_filter
        self.shuffle_buffer = shuffle_buffer
        self.kwargs = kwargs
        self.num_process = 1 #change later by mpi
        self.num_local_process = 1 #change later by mpi
        self.rank = 0 #change later by mpi
        self.local_rank = 0 #change later by mpi
        self.decode_type = None #change later by tfrecord config
        self.input_names = [] #change later by network
        self.output_names = [] #change later by network
        self.compression = None
        self.block_length = {"train": 2, "eval":1}

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

    def _input_source(self, mode):
        """
        Prepares data from TFRecords for input into the model

        Args:
            mode: Mode for training ("train", "eval" or "both")

        Returns:
            Dataset object containing the batch of tensors to be ingested by the model
        """
        filenames = self.file_names[mode]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if mode == "train":
            dataset = dataset.shard(self.num_local_process, self.local_rank)
            dataset = dataset.shuffle(len(filenames))
        dataset = dataset.interleave(lambda dataset: tf.data.TFRecordDataset(dataset, compression_type=self.compression), cycle_length=self.num_subprocess, block_length=self.block_length[mode])
        if self.shuffle_buffer == "auto":
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(min(10000, self.num_examples[mode])))
        else:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(self.shuffle_buffer))
        dataset = dataset.map(lambda dataset: self.read_and_decode(dataset))
        if self.data_filter is not None and self.data_filter.mode in [mode, "both"]:
            dataset = dataset.filter(lambda dataset: self.data_filter.predicate_fn(dataset))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda dataset: self._preprocess_fn(dataset, mode), self.batch_size))
        dataset = dataset.map(lambda dataset: self.final_transform(dataset))
        dataset = dataset.map(lambda dataset: self._split_in_out(dataset))
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
                if isinstance(preprocess_obj, StaticAugmentation):
                    if preprocess_obj.mode == mode or preprocess_obj.mode == "both":
                        preprocess_obj.height = preprocess_data.get_shape()[0].value
                        preprocess_obj.width = preprocess_data.get_shape()[1].value
                        preprocess_obj.feature_name = feature_name
                        if preprocess_obj not in randomized_list:
                            preprocess_obj.decoded_data = decoded_data
                            preprocess_obj.setup()
                            randomized_list.append(preprocess_obj)
                        preprocess_data = preprocess_obj.transform(preprocess_data)
                else:
                    preprocess_obj.feature_name = feature_name
                    preprocess_data = preprocess_obj.transform(preprocess_data, decoded_data)
            preprocessed_data[feature_name] = preprocess_data
        return preprocessed_data

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

    def _split_in_out(self, preprocessed_data):
        """
        Assigns features as input or output to the model

        Args:
            preprocessed_data: Dictionary of tensors that the model takes

        Returns:
            Input and output dictionaries of tensors that the model takes
        """
        if len(self.input_names) == 0 or len(self.output_names) == 0:
            return preprocessed_data
        else:
            features = {}
            targets = {}
            for feature in preprocessed_data.keys():
                if feature in self.input_names:
                    features[feature] = preprocessed_data[feature]
                if feature in self.output_names:
                    targets[feature] = preprocessed_data[feature]
            return features, targets

    def read_and_decode(self, dataset):
        """
        Reads and decodes the string data from TFRecords

        Args:
            dataset: Dataset consisting of encoded data from TFRecords

        Returns:
            Dictionary of decoded data

        """
        decoded_data = {}
        all_data = tf.parse_single_example(dataset, features=self.keys_to_features)
        for feature in self.feature_name:
            data = all_data[feature]
            if "string" in str(data.dtype) and "string" not in self.decode_type[feature]:
                data = tf.decode_raw(data, convert_tf_dtype(self.decode_type[feature]))
            if "int" in str(data.dtype):
                data = tf.cast(data, tf.int32)
            elif self.decode_type[feature] == "string":
                data = data
            else:
                data = tf.cast(data, tf.float32)
            decoded_data[feature] = data
        return decoded_data

    def show_batches(self, mode="train", inputs=None, num_batches=1):
        """
        Shows batches of tensor data in numpy

        Args:
            mode: Mode for training ("train", "eval" or "both")
            inputs: Directory for saving TFRecords
            num_batches: Number of batches to show

        Returns:
            A dictionary containing the batches numpy data with corresponding keys
        """
        np_data = {}
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
        dataset = self._input_source(mode)
        iterator = dataset.make_one_shot_iterator()
        data_tensor = iterator.get_next()
        for key in data_tensor.keys():
            np_data[key] = []
        with tf.Session() as sess:
            for i in range(num_batches):
                np_batch = sess.run(data_tensor)
                for key in data_tensor.keys():
                    np_data[key].append(np_batch[key])
        return np_data
