from fastestimator.pipeline.augmentation import AbstractAugmentation
from fastestimator.util.tfrecord import TFRecorder, get_features
from fastestimator.util.util import convert_tf_dtype
import tensorflow as tf
import multiprocessing
import numpy as np
import time
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
        data_filter: Filtering to be performed on the corresponding features in the form of an object from the
            Filter class
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
                 **kwargs):
        self.batch_size = batch_size
        self.train_data = train_data
        self.feature_name = feature_name
        self.transform_train = transform_train
        self.transform_dataset = transform_dataset
        self.validation_data = validation_data
        self.data_filter = data_filter
        self.kwargs = kwargs
        self.num_process = 1 #change later by mpi
        self.num_local_process = 1 #change later by mpi
        self.rank = 0 #change later by mpi
        self.local_rank = 0 #change later by mpi
        self.decode_type = None #change later by tfrecord config
        self.feature_shape = None #change later by tfrecord config
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
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if mode == "train":
            dataset = dataset.shard(self.num_local_process, self.local_rank)
            dataset = dataset.shuffle(len(filenames))
        dataset = dataset.interleave(lambda dataset: tf.data.TFRecordDataset(dataset, compression_type=self.compression), cycle_length=self.num_subprocess, block_length=self.block_length[mode])
        if mode == "train":
            dataset = dataset.shuffle(min(10000, self.num_examples[mode]))
            dataset = dataset.repeat()
        dataset = dataset.map(lambda dataset: self.read_and_decode(dataset), num_parallel_calls=self.num_subprocess)
        if self.data_filter is not None and self.data_filter.mode in [mode, "both"]:
            dataset = dataset.filter(lambda dataset: self.data_filter.predicate_fn(dataset))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _transform_batch(self, batch_data, mode):
        # single_example = dict()
        # for k in batch_data.keys():
        #     single_example[k] = None

        for feature_idx, feature_name in enumerate(batch_data.keys()):
            feature = batch_data[feature_name].numpy()
            transform_list = self.transform_train[feature_idx]
            # transform each example
            # \TODO(jp) how to add decoded data here?
            for i in range(self.batch_size):
                # for k in batch_data.keys():
                #     single_example[k] = batch_data[k][i,...]
                # def _get_single_example():
                #     feature_slice = dict()
                #     for k in batch_data.keys():
                #         feature_slice[k] = batch_data[k][i, ...]
                #     return feature_slice
                # single_example = _get_single_example()
                for process_obj in transform_list:
                    # process_obj.decoded_data = single_example
                    if isinstance(process_obj, AbstractAugmentation):
                        process_obj.setup()
                    feature[i, ...] = process_obj.transform(feature[i, ...])
            batch_data[feature_name] = feature
        return batch_data

    def _input_source(self, mode):
        """Package the data from tfrecord to model
        
        Args:
            mode (str): mode for current pipeline ("train", "eval" or "both")
        
        Returns:
            Iterator: An iterator that can provide a streaming of processed data
        """
        dataset = self._input_stream(mode)
        # init aug/preprocess objs
        for feature_idx, feature_name in enumerate(self.feature_name):
            transform_list = self.transform_train[feature_idx]
            for process_obj in transform_list:
                process_obj.feature_name = feature_name
                if isinstance(process_obj, AbstractAugmentation):
                    if process_obj.mode == mode or process_obj.mode == "both":
                        process_obj.width = self.feature_shape[feature_name][1]
                        process_obj.height = self.feature_shape[feature_name][2]

        for batch_data in dataset:
            yield self._transform_batch(batch_data, mode)

    def _combine_dict(self, dict_list):
        """combine same key of multiple dictionaries list into one dictinoary
        
        Args:
            dict_list (list): list of dictionaries
        
        Returns:
            dict: combined dictionary
        """
        combined_batch = {}
        for feature in dict_list[0].keys():
            combined_batch[feature] = np.array(list(d[feature] for d in dict_list))
        return combined_batch
        
    def _get_dict_slice(self, dictionary, index):
        """slice a dictionary for each key and return the sliced dictionary
        
        Args:
            dictionary (dict): original dictionary
            index (int): slice index
        
        Returns:
            dict: sliced dictionary with same key as original dictionary
        """
        feature_slice = dict()
        keys = dictionary.keys()
        for key in keys:
            feature_slice[key] = dictionary[key][index]
        return feature_slice

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
            A dictionary containing the batches numpy data with corresponding keys
        """
        np_data = []
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
        for example in dataset.take(num_batches):
            for key in example.keys():
                example[key] = example[key].numpy()
            np_data.append(example)
        return np_data

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
        it = self._input_source(mode)
        start = time.time()
        for i, _ in enumerate(it):
            if i % log_interval == 0 and i >0:
                duration = time.time() - start
                example_per_sec = log_interval * self.batch_size / duration
                print("FastEstimator: Pipeline Preprocessing Example/sec %f" % example_per_sec)
                start = time.time()
