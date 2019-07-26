from fastestimator.util.op import flatten_operation, get_op_from_mode, verify_ops
from fastestimator.util.util import convert_tf_dtype
import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import time
import os

class RecordWriter:
    """
    Class for creating TFRecords from numpy data or csv file containing paths to data on disk

    Args:
        train_data: Training dataset in the form of dictionary containing numpy data, or csv file (with file
            paths or data)
        feature_name: List of strings representing the feature names in the data (headers in csv, keys in dictionary
            or features in TFRecords)
        transform_dataset: List of lists of numpy transformations to be performed sequentially  on the raw data
            before the TFRecords are made.
        validation_data:  Validation data in the form of dictionary containing numpy data, or csv file, or fraction
            of training data to be sequestered for validation during training.
        create_patch: Whether to create multiple records from single example.
        max_tfrecord_mb: Maximum space to be occupied by one TFRecord.
        compression: tfrecords compression type, one of None, 'GZIP' or 'ZLIB'.
    """
    def __init__(self, train_data, validation_data=None, ops=None, write_feature=None, create_patch=False, max_tfrecord_mb=300, compression=None):
        self.train_data = train_data
        self.validation_data = validation_data
        self.ops = ops
        self.write_feature = write_feature
        self.create_patch = create_patch
        self.max_tfrecord_mb = max_tfrecord_mb
        self.compression = compression
        self.num_process = 1
        self.rank = 0
        self.local_rank = 0
        self.num_subprocess = mp.cpu_count()
        self.compression_option = tf.io.TFRecordOptions(compression_type=compression)
        self.num_example_csv = {}
        self.mode_ops = {}
        self.final_feature_name = {}
        self.mb_per_csv_example = {}
        self.feature_dtype = {}
        self.feature_shape = {}
        self.feature_name = {}
        self.num_example_record = {}

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def create_tfrecord(self, save_dir):
        self._verify_input()
        self._prepare_train()
        if self.validation_data:
            self._prepare_validation()
            self._get_feature_info(self.validation_data, "eval")
        self._get_feature_info(self.train_data, "train")
        self._prepare_savepath(save_dir)
        self.num_example_record["train"] = self._write_tfrecord_parallel(self.train_data, mode="train")
        if self.validation_data:
            self.num_example_record["eval"] = self._write_tfrecord_parallel(self.validation_data, mode="eval")
        self._write_json_summary()

    def _get_feature_info(self, dictionary, mode):
        feature = self._transform_one_slice(dictionary=dictionary, index=0, mode=mode)
        if self.write_feature is None:
            self.feature_name[mode] = list(feature.keys())
        elif isinstance(self.write_feature, dict):
            self.feature_name[mode] = self.write_feature[mode]
        else:
            self.feature_name[mode] = self.write_feature
        self.mb_per_csv_example[mode] = 0
        self.feature_dtype[mode] = {}
        self.feature_shape[mode] = {}
        for name in self.feature_name[mode]:
            data = np.asarray(feature[name])
            self.mb_per_csv_example[mode] += data.nbytes / 1e6
            dtype = str(data.dtype)
            if "<U" in dtype:
                dtype = "str"
            self.feature_dtype[mode][name] = dtype
            if data.size == 1 or max(data.shape) == np.prod(data.shape):
                self.feature_shape[mode][name] = [-1]
            else:
                self.feature_shape[mode][name] = data.shape
        return 

    def _write_tfrecord_parallel(self, dictionary, mode):
        num_example_list = []
        processes = []
        queue = mp.Queue()
        num_example_process = self.num_example_csv[mode] // self.num_process
        process_start = self.rank * num_example_process
        if self.rank == (self.num_process -1):
            process_end = self.num_example_csv[mode]
        else:
            process_end = process_start + num_example_process
        num_example_process = process_end - process_start
        num_files_subprocess = int(np.ceil(num_example_process * self.mb_per_csv_example[mode] / self.max_tfrecord_mb / self.num_subprocess))
        file_idx_start_process = self.rank * num_files_subprocess * self.num_subprocess
        num_example_subprocess_remain = num_example_process % self.num_subprocess
        serial_start = process_start
        for i in range(self.num_subprocess):
            file_idx_start_subprocess = i * num_files_subprocess + file_idx_start_process
            if i < num_example_subprocess_remain:
                num_example_subprocess = num_example_process // self.num_subprocess + 1
            else:
                num_example_subprocess = num_example_process // self.num_subprocess
            serial_end = serial_start + num_example_subprocess
            processes.append(mp.Process(target=self._write_tfrecord_serial, args=(dictionary, serial_start, serial_end, num_files_subprocess, file_idx_start_subprocess, mode, queue)))
            serial_start += num_example_subprocess
        for p in processes:
            p.start()
        for _ in range(self.num_subprocess):
            num_example_list.extend(queue.get(block=True))
        for p in processes:
            p.join()
        return num_example_list

    def _write_tfrecord_serial(self, dictionary, serial_start, serial_end, num_files_subprocess, file_idx_start, mode, queue):
        num_example_list = []
        num_csv_example_per_file = (serial_end - serial_start) // num_files_subprocess
        show_progress = serial_start == 0
        file_start = serial_start
        file_end = file_start + num_csv_example_per_file
        for i in range(num_files_subprocess):
            file_idx = i + file_idx_start
            filename = mode + str(file_idx) + ".tfrecord"
            if i == (num_files_subprocess - 1):
                file_end = serial_end
            num_example_list.append(self._write_single_file(dictionary, filename, file_start, file_end, serial_start, serial_end, show_progress, mode))
            file_start += num_csv_example_per_file
            file_end += num_csv_example_per_file
        queue.put(num_example_list)

    def _write_single_file(self, dictionary, filename, file_start, file_end, serial_start, serial_end, show_progress, mode):
        goal_number = serial_end - serial_start
        logging_interval = max(goal_number // 20, 1)
        num_example = 0
        with tf.io.TFRecordWriter(os.path.join(self.save_dir, filename), options= self.compression_option) as writer:
            for i in range(file_start, file_end):
                if i == file_start and show_progress:
                    time_start = time.time()
                    example_start = num_example
                if (i - serial_start) % logging_interval == 0 and show_progress:
                    if i == 0:
                        record_per_sec = 0.0
                    else:
                        record_per_sec = (num_example - example_start) * self.num_process * self.num_subprocess/(time.time() - time_start)
                    print("FastEstimator: Converting %s TFRecords %.1f%%, Speed: %.2f record/sec" % (mode.capitalize(), (i - serial_start)/goal_number*100, record_per_sec))
                feature = self._transform_one_slice(dictionary, i, mode=mode)
                if self.create_patch:
                    num_patches = self._verify_dict(feature, mode)
                    for j in range(num_patches):
                        feature_patch = self._get_dict_slice(feature, j)
                        self._write_single_example(feature_patch, writer, mode)
                        num_example += 1
                else:
                    self._write_single_example(feature, writer, mode)
                    num_example += 1
        return filename, num_example

    def _transform_one_slice(self, dictionary, index, mode):
        feature = self._get_dict_slice(dictionary, index)
        if self.ops:
            feature = self._preprocessing(feature, mode)
        return feature

    def _write_single_example(self, dictionary, writer, mode):
        feature_tfrecord = {}
        for name in self.feature_name[mode]:
            data = np.array(dictionary[name]).astype(self.feature_dtype[mode][name])
            feature_tfrecord[name] = self._bytes_feature(data.tostring())
        example = tf.train.Example(features=tf.train.Features(feature=feature_tfrecord))
        writer.write(example.SerializeToString())

    def _get_dict_slice(self, dictionary, index):
        feature_slice = dict()
        keys = dictionary.keys()
        for key in keys:
            feature_slice[key] = dictionary[key][index]
        return feature_slice

    def _preprocessing(self, feature, mode):
        for op in self.mode_ops[mode]:
            if op.inputs:
                data = self._get_inputs_from_key(feature, op.inputs)
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

    def _verify_dict(self, dictionary, mode=None):
        if mode is None:
            feature_name = dictionary.keys()
        else:
            feature_name = self.feature_name[mode]
        num_example_list = []
        for key in feature_name:
            feature_data = dictionary[key]
            if type(feature_data) is list:
                num_example_list.append(len(feature_data))
            elif type(feature_data) is np.ndarray:
                num_example_list.append(feature_data.shape[0])
            else:
                raise ValueError("the feature only supports list or numpy array")
        assert len(set(num_example_list)) == 1, "features should have the same number of examples"
        return set(num_example_list).pop()

    def _prepare_train(self):
        if type(self.train_data) is str:
            df = pd.read_csv(self.train_data)
            self.train_data = df.to_dict('list')
        self.num_example_csv["train"] = self._verify_dict(self.train_data)
        self._check_ops("train")

    def _prepare_validation(self):
        if type(self.validation_data) is float:
            num_example_takeout = int(self.validation_data * self.num_example_csv["train"])
            train_idx = range(self.num_example_csv["train"])
            eval_idx = np.random.choice(train_idx, num_example_takeout, replace=False)
            train_idx = np.delete(train_idx, eval_idx)
            self.validation_data = {}
            for key in self.train_data.keys():
                total_data = self.train_data[key]
                if type(total_data) is list:
                    self.train_data[key] = [total_data[x] for x in train_idx]
                    self.validation_data[key] = [total_data[x] for x in eval_idx]
                else:
                    self.train_data[key] = total_data[train_idx]
                    self.validation_data[key] = total_data[eval_idx]
            self.num_example_csv["train"] = self.num_example_csv["train"] - num_example_takeout
        elif type(self.validation_data) is str:
            df = pd.read_csv(self.validation_data)
            self.validation_data = df.to_dict('list')
        self.num_example_csv["eval"] = self._verify_dict(self.validation_data)
        self._check_ops("eval")

    def _check_ops(self, mode):
        if self.ops:
            self.mode_ops[mode] = get_op_from_mode(self.ops, mode)
            if len(self.mode_ops[mode]) > 0:
                verify_ops(self.mode_ops[mode], "RecordWriter")

    def _prepare_savepath(self, save_dir):
        self.save_dir = save_dir
        if self.local_rank == 0:
            print("FastEstimator: Saving tfrecord to %s" % self.save_dir)
            assert not os.path.exists(self.save_dir), "data path already exits: %s" % self.save_dir
            os.makedirs(self.save_dir)
        if self.num_process > 1:
            import horovod.tensorflow.keras as hvd
            hvd.allreduce([0], name="Barrier")

    def _verify_input(self):
        assert type(self.train_data) is dict or self.train_data.endswith(".csv"), "data should either be a dictionary or a csv file"
        if self.validation_data:
            assert type(self.validation_data) in [dict, float] or self.validation_data.endswith(".csv"), "validation data supports partition ratio (float), csv file or dictionary"
        if self.write_feature:
            assert isinstance(self.write_feature, (list, dict)), "write_feature must be either list or dictionary"
        if self.ops:
            self.ops = flatten_operation(self.ops)

    def _write_json_summary(self):
        summary = {"feature_dtype": self.feature_dtype, "feature_shape": self.feature_shape, "file_names": {}, "num_records":{}}
        train_files, num_train_examples = zip(*self.num_example_record["train"])
        summary["file_names"]["train"] = list(train_files)
        summary["num_records"]["train"] = list(num_train_examples)
        if self.compression:
            summary["compression"] = self.compression
        if self.validation_data:
            eval_files, num_eval_examples = zip(*self.num_example_record["eval"])
            summary["file_names"]["eval"] = list(eval_files)
            summary["num_records"]["eval"] = list(num_eval_examples)
        file_name = "summary%d.json" % self.rank
        with open(os.path.join(self.save_dir, file_name), 'w') as fp:
            json.dump(summary, fp, indent=4)