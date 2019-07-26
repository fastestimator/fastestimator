import json
import multiprocessing as mp
import os
import time
import numpy as np
import tensorflow as tf
from fastestimator.util.op import flatten_operation, get_op_from_mode, verify_ops
from fastestimator.pipeline.augmentation import TensorAugmentation
from fastestimator.pipeline.filter import Filter
from fastestimator.util.tfrecord import get_features
from fastestimator.util.util import convert_tf_dtype
from fastestimator.record.record import RecordWriter

class Pipeline:
    def __init__(self,
                 batch_size,
                 data=None,
                 ops=None,
                 read_feature=None,
                 data_filter=None,
                 padded_batch=False,
                 multi_patch=False):

        self.batch_size = batch_size
        self.data = data
        self.ops = ops
        self.read_feature = read_feature
        self.data_filter = data_filter
        self.padded_batch = padded_batch
        self.multi_patch = multi_patch
        self.feature_dtype = None #change later by tfrecord config
        self.feature_shape = None #change later by tfrecord config
        self.compression = None
        self.num_local_process = 1
        self.local_rank = 0
        self.feature_name = {}
        self.all_features = {}
        self.mode_ops = {}
        self.num_subprocess = mp.cpu_count()//self.num_local_process
        self._verify_input()

    def _verify_input(self):
        assert isinstance(self.data, (dict, RecordWriter)), "data should either be a RecordWriter instance or a dictionary"
        if self.read_feature:
            assert isinstance(self.read_feature, (list, dict)), "write_feature must be either list or dictionary"
        if self.ops:
            self.ops = flatten_operation(self.ops)

    def _prepare(self, inputs):
        if isinstance(self.data, RecordWriter):
            if inputs is None:
                raise ValueError("Must specify the data path of tfrecords")
            if os.path.exists(inputs):
                print("FastEstimator: Using existing tfrecords in {}".format(inputs))
            else:
                self.data.create_tfrecord(save_dir=inputs)
            self._get_tfrecord_config(inputs)
        else:
            self.all_features = self.data
        self._get_feature_name(mode="train")
        self._check_ops(mode="train")
        if "eval" in self.all_features:
            self._get_feature_name(mode="eval")
            self._check_ops(mode="eval")

    def _get_feature_name(self, mode):
        if self.read_feature is None:
            self.feature_name[mode] = list(self.all_features[mode].keys())
        elif isinstance(self.read_feature, dict):
            self.feature_name[mode] = self.read_feature[mode]
        else:
            self.feature_name[mode] = self.read_feature

    def _check_ops(self, mode):
        if self.ops:
            self.mode_ops[mode] = get_op_from_mode(self.ops, mode)
            if len(self.mode_ops[mode]) > 0:
                verify_ops(self.mode_ops[mode], "Pipeline")

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
        self.num_records = {"train": 0, "eval": 0}
        for json_file in json_files:
            summary = json.load(open(json_file, 'r'))
            if self.feature_dtype is None:
                self.feature_dtype = summary["feature_dtype"]
            if self.feature_shape is None:
                self.feature_shape = summary["feature_shape"]
            if "compression" in summary:
                self.compression = summary["compression"]
            if "eval" in summary["file_names"]:
                self.file_names["eval"].extend([os.path.join(data_dir, f) for f in summary["file_names"]["eval"]])
                self.num_records["eval"] += np.sum(summary["num_records"]["eval"])
                if not "eval" in self.all_features:
                    self.all_features["eval"] = get_features(self.file_names["eval"][0], compression=self.compression)
            self.file_names["train"].extend([os.path.join(data_dir, f) for f in summary["file_names"]["train"]])
            self.num_records["train"] += np.sum(summary["num_records"]["train"])
            if not "train" in self.all_features:
                self.all_features["train"] = get_features(self.file_names["train"][0], compression=self.compression)
        assert len(self.file_names["train"]) >= self.num_local_process, "number of training file per local process should at least be 1"
        if self.local_rank == 0:
            print("FastEstimator: Found %d examples for training and %d for validation in %s" %(self.num_records["train"], self.num_records["eval"], data_dir))

    def _input_stream(self, mode):
        """
        Prepares data from TFRecords for streaming input

        Args:
            mode: Mode for current pipeline ("train", "eval")

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
            dataset = dataset.shuffle(min(10000, self.num_records[mode]))
            dataset = dataset.repeat()
        else:
            dataset = tf.data.TFRecordDataset(filenames, compression_type=self.compression)
        # reading and decoding
        dataset = dataset.map(lambda dataset: self.read_and_decode(dataset, mode), num_parallel_calls=self.num_subprocess)
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

    def _preprocess_fn(self, feature, mode):
        """
        Preprocessing performed on the tensor data in features in the order specified in the transform_train list
        Args:
            decoded_data: dataset object containing a dictionary of tensors
            mode: Mode for training ("train", "eval" or "both")
        Returns:
            Dictionary containing the preprocessed data in the form of a dictionary of tensors
        """
        randomized_list = []
        for op in self.mode_ops[mode]:
            if op.inputs:
                data = self._get_inputs_from_key(feature, op.inputs)
            if isinstance(op, TensorAugmentation):
                op.height = data.get_shape()[0]
                op.width = data.get_shape()[1]
                if op not in randomized_list:
                    op.setup()
                    randomized_list.append(op)
            data = op.forward(data)
            if op.outputs:
                feature = self._write_outputs_to_key(feature, data, op.outputs)
        return feature

    def _write_outputs_to_key(self, feature, outputs_data, outputs_key):
        if isinstance(outputs_key, str):
            feature[outputs_key] = outputs_data
        else:
            for key, data in zip(outputs_key, outputs_data):
                feature[key] = data
        return feature

    def _get_inputs_from_key(self, feature, inputs_key):
        if isinstance(inputs_key, list):
            data = [feature[key] for key in inputs_key]
        elif isinstance(inputs_key, tuple):
            data = tuple([feature[key] for key in inputs_key])
        else:
            data = feature[inputs_key]
        return data

    def read_and_decode(self, dataset, mode):
        """
        Reads and decodes the string data from TFRecords

        Args:
            dataset: Dataset consisting of encoded data from TFRecords

        Returns:
            Dictionary of decoded data

        """
        decoded_data = {}
        all_data = tf.io.parse_single_example(dataset, features=self.all_features[mode])
        for feature in self.feature_name[mode]:
            data = all_data[feature]
            if "str" in str(data.dtype) and "str" not in self.feature_dtype[mode][feature]:
                data = tf.io.decode_raw(data, convert_tf_dtype(self.feature_dtype[mode][feature]))
                data = tf.reshape(data, self.feature_shape[mode][feature])
                print(self.feature_shape[mode][feature])
            if "int" in str(data.dtype):
                data = tf.cast(data, tf.int32)
            elif not self.decode_type[feature] == "str":
                data = tf.cast(data, tf.float32)
            decoded_data[feature] = data
        return decoded_data

    def show_results(self, inputs=None, mode="train", num_steps=1):
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
        self._prepare(inputs=inputs)
        dataset = self._input_stream(mode)
        for i, example in enumerate(dataset.take(num_steps)):
            data.append(example)
        return data

    def benchmark(self, inputs=None, mode="train", num_steps= 500, log_interval= 100):
        """
        benchmark the pipeline processing speed during training

        Args:
            inputs: Directory for saving TFRecords
            mode: Mode for training ("train", "eval" or "both")
        """
        self._prepare(inputs=inputs)
        dataset = self._input_stream(mode)
        start = time.time()
        for i, _ in enumerate(dataset.take(num_steps)):
            if i % log_interval == 0 and i >0:
                duration = time.time() - start
                example_per_sec = log_interval * self.batch_size / duration
                print("FastEstimator: Pipeline Preprocessing Example/sec %f" % example_per_sec)
                start = time.time()
