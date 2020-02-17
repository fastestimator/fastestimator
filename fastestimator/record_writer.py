# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility for writing TFRecords."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf

from fastestimator.op import get_inputs_by_op, get_op_from_mode, verify_ops, write_outputs_by_key


class RecordWriter:
    """Write data into TFRecords.

    This class can handle unpaired features. For example, in cycle-gan the hourse and zebra images are unpaired, which
    means during training you do not have one-to-one correspondance between hourse image and zebra image. When the
    `RecordWriter` instance is sent to `Pipeline` create random pairs between hourse and zebra images. See the cycle-gan
    example in apphub directory.

    Args:
        train_data (Union[dict, str]): A `dict` that contains train data or a CSV file path. For the CSV file, the
            column header will be used as feature name. Under each column in the CSV file the paths to train data should
            be provided.
        save_dir (str): The directory to save the TFRecords.
        validation_data (Union[dict, str, float], optional): A `dict` that contains validation data, a CSV file path, or
            a `float` that is between 0 and 1. For the CSV file, the column header will be used as feature name. Under
            each column in the CSV file the paths to validation data should be provided. When this argument is a
            `float`, `RecordWriter` will reserve `validation_data` fraction of the `train_data` as validation data.
            Defaults to None.
        ops (obj, optional): Transformation operations before TFRecords creation. Defaults to None.
        write_feature (str, optional): Users can specify what features they want to write to TFRecords. Defaults to
            None.
        expand_dims (bool, optional): When set to `True`, the first dimension of each feature will be used as batch
            dimension. `RecordWriter` will split the batch into single examples and write one example at a time into
            TFRecord. Defaults to False.
        max_record_size_mb (int, optional): Maximum size of single TFRecord file. Defaults to 300 MB.
        compression (str, optional): Compression type can be `"GZIP"`, `"ZLIB"`, or `""` (no compression). Defaults to
            None.
    """
    def __init__(self,
                 train_data,
                 save_dir,
                 validation_data=None,
                 ops=None,
                 write_feature=None,
                 expand_dims=False,
                 max_record_size_mb=300,
                 compression=None):
        self.train_data = train_data
        self.save_dir = save_dir
        self.validation_data = validation_data
        self.ops = ops
        self.write_feature = write_feature
        self.expand_dims = expand_dims
        self.max_record_size_mb = max_record_size_mb
        self.compression = compression
        self.num_process = os.cpu_count() or 1
        self.compression_option = tf.io.TFRecordOptions(compression_type=compression)
        self.global_file_idx = {"train": 0, "eval": 0}
        self.global_feature_key = {"train": [], "eval": []}
        self.feature_set_idx = 0
        self._verify_inputs()

        self.num_example_csv, self.num_example_record, self.mb_per_csv_example, self.mb_per_record_example = \
            {}, {}, {}, {}
        self.mode_ops, self.feature_name, self.feature_dtype, self.feature_shape = {}, {}, {}, {}
        self.train_data_local, self.validation_data_local, self.write_feature_local, self.ops_local = {}, {}, {}, {}

    def _verify_inputs(self):
        if any(isinstance(inp, tuple) for inp in [self.train_data, self.validation_data, self.ops, self.write_feature]):
            num_unpaired_feature_sets = [len(self.train_data)]
            if self.validation_data:
                assert isinstance(self.validation_data, tuple), \
                    "validation data must be tuple when creating unpaired feature set"
                num_unpaired_feature_sets.append(len(self.validation_data))
            else:
                self.validation_data = [None] * len(self.train_data)
            if self.ops:
                assert isinstance(self.ops, tuple), "operation must be tuple when creating unpaired feature set"
                num_unpaired_feature_sets.append(len(self.ops))
            else:
                self.ops = [None] * len(self.train_data)
            if self.write_feature:
                assert isinstance(self.write_feature,
                                  tuple), "write_feature must be tuple when creating unpaired feature set"
                num_unpaired_feature_sets.append((len(self.write_feature)))
            else:
                self.write_feature = [None] * len(self.train_data)
            assert len(set(
                num_unpaired_feature_sets)) == 1, "tuple length should be consistent when creating unpaired feature set"
            self.train_data, self.validation_data, self.ops, self.write_feature = list(self.train_data), list(
                self.validation_data), list(self.ops), list(self.write_feature)
        else:
            self.train_data, self.validation_data, self.ops, self.write_feature = \
                [self.train_data], [self.validation_data], [self.ops], [self.write_feature]
        for idx in range(len(self.train_data)):
            assert isinstance(self.train_data[idx], dict) or self.train_data[idx].endswith(
                ".csv"), "train data should either be a dictionary or a csv path"
            if self.validation_data[idx]:
                assert isinstance(self.validation_data[idx], (dict, float)) or self.validation_data[idx].endswith(
                    ".csv"), "validation data supports partition ratio (float), csv file or dictionary"
            if self.write_feature[idx]:
                assert isinstance(self.write_feature[idx],
                                  (list, dict)), "write_feature must be either list or dictionary"
            if self.ops[idx]:
                if not isinstance(self.ops[idx], list):
                    self.ops[idx] = [self.ops[idx]]
            else:
                self.ops[idx] = []

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def write(self, save_dir=None):
        """Write TFRecods in parallel. Number of processes is set to number of CPU cores."""
        if not save_dir:
            save_dir = self.save_dir
        self._prepare_savepath(save_dir)
        for train_data, validation_data, write_feature, ops in zip(self.train_data, self.validation_data,
                                                                   self.write_feature, self.ops):
            self._create_record_local(train_data, validation_data, write_feature, ops)
            self.feature_set_idx += 1

    def _create_record_local(self, train_data, validation_data, write_feature, ops):
        self.num_example_csv, self.num_example_record, self.mb_per_csv_example, self.mb_per_record_example = \
            {}, {}, {}, {}
        self.mode_ops, self.feature_name, self.feature_dtype, self.feature_shape = {}, {}, {}, {}
        self.train_data_local, self.validation_data_local, self.write_feature_local, self.ops_local = \
            train_data, validation_data, write_feature, ops
        self._prepare_train()
        self._get_feature_info(self.train_data_local, "train")
        if validation_data:
            self._prepare_validation()
            self._get_feature_info(self.validation_data_local, "eval")
        self.num_example_record["train"] = self._write_tfrecord_parallel(self.train_data_local, mode="train")
        self._write_json_summary("train")
        if validation_data:
            self.num_example_record["eval"] = self._write_tfrecord_parallel(self.validation_data_local, mode="eval")
            self._write_json_summary("eval")

    def _get_feature_info(self, dictionary, mode):
        feature = self._transform_one_slice(dictionary=dictionary, index=0, mode=mode)
        if self.write_feature_local is None:
            self.feature_name[mode] = list(feature.keys())
        elif isinstance(self.write_feature_local, dict):
            self.feature_name[mode] = self.write_feature_local[mode]
        else:
            self.feature_name[mode] = self.write_feature_local
        self.global_feature_key[mode].extend(self.feature_name[mode])
        assert len(set(self.global_feature_key[mode])) == len(
            self.global_feature_key[mode]), "found duplicate key in feature name during {}: {}".format(
            mode, self.global_feature_key[mode])
        self.mb_per_csv_example[mode] = 0
        self.mb_per_record_example[mode] = 0
        self.feature_dtype[mode] = {}
        self.feature_shape[mode] = {}
        for key in self.feature_name[mode]:
            data = np.asarray(feature[key])
            self.mb_per_csv_example[mode] += data.nbytes / 1e6
            if self.expand_dims:
                data = data[0]
            self.mb_per_record_example[mode] += data.nbytes / 1e6
            dtype = str(data.dtype)
            if "<U" in dtype:
                dtype = "str"
            self.feature_dtype[mode][key] = dtype
            if data.size == 1:
                self.feature_shape[mode][key] = []
            elif max(data.shape) == np.prod(data.shape):
                self.feature_shape[mode][key] = [-1]
            else:
                self.feature_shape[mode][key] = data.shape

    def _write_tfrecord_parallel(self, dictionary, mode):
        num_example_list = []
        feature_shape_list = []
        num_files_process = int(
            np.ceil(self.num_example_csv[mode] * self.mb_per_csv_example[mode] / self.max_record_size_mb /
                    self.num_process))
        num_example_process_remain = self.num_example_csv[mode] % self.num_process
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_process) as executor:
            serial_start = 0
            file_idx_start = 0
            for i in range(self.num_process):
                if i < num_example_process_remain:
                    num_example_process = self.num_example_csv[mode] // self.num_process + 1
                else:
                    num_example_process = self.num_example_csv[mode] // self.num_process
                serial_end = serial_start + num_example_process
                futures.append(
                    executor.submit(self._write_tfrecord_serial,
                                    dictionary,
                                    serial_start,
                                    serial_end,
                                    num_files_process,
                                    file_idx_start,
                                    mode))
                serial_start += num_example_process
                file_idx_start += num_files_process
        for future in futures:
            result = future.result()
            num_example_list.extend(result[0])
            feature_shape_list.append(result[1])
        self._reconfirm_shape(feature_shape_list, mode)
        return num_example_list

    def _reconfirm_shape(self, feature_shape_list, mode):
        feature_shape = self.feature_shape[mode]
        for new_feature_shape in feature_shape_list:
            for key in feature_shape:
                if new_feature_shape[key] == [-1] and feature_shape[key] == []:
                    feature_shape[key] = [-1]
        self.feature_shape[mode] = feature_shape

    def _write_tfrecord_serial(self, dictionary, serial_start, serial_end, num_files_process, file_idx_start, mode):
        num_example_list = []
        num_csv_example_per_file = (serial_end - serial_start) // num_files_process
        show_progress = serial_start == 0
        file_start = serial_start
        file_end = file_start + num_csv_example_per_file
        for i in range(num_files_process):
            file_idx = i + file_idx_start + self.global_file_idx[mode]
            filename = mode + str(file_idx) + ".tfrecord"
            if i == (num_files_process - 1):
                file_end = serial_end
            num_example_list.append(
                self._write_single_file(dictionary,
                                        filename,
                                        file_start,
                                        file_end,
                                        serial_start,
                                        serial_end,
                                        show_progress,
                                        mode))
            file_start += num_csv_example_per_file
            file_end += num_csv_example_per_file
        return num_example_list, self.feature_shape[mode]

    def _write_single_file(self,
                           dictionary,
                           filename,
                           file_start,
                           file_end,
                           serial_start,
                           serial_end,
                           show_progress,
                           mode):
        goal_number = serial_end - serial_start
        logging_interval = max(goal_number // 20, 1)
        num_example = 0
        with tf.io.TFRecordWriter(os.path.join(self.save_dir, filename), options=self.compression_option) as writer:
            for i in range(file_start, file_end):
                if i == file_start and show_progress:
                    time_start = time.perf_counter()
                    example_start = num_example
                if (i - serial_start) % logging_interval == 0 and show_progress:
                    if i == 0:
                        record_per_sec = 0.0
                    else:
                        record_per_sec = (num_example - example_start) * self.num_process / (time.perf_counter() -
                                                                                             time_start)
                    print("FastEstimator: Converting %s TFRecords %.1f%%, Speed: %.2f record/sec" %
                          (mode.capitalize(), (i - serial_start) / goal_number * 100, record_per_sec))
                feature = self._transform_one_slice(dictionary, i, mode=mode)
                if self.expand_dims:
                    num_patches = self._verify_dict(feature, mode)
                    for j in range(num_patches):
                        feature_patch = self._get_dict_slice(feature, j, keys=self.feature_name[mode])
                        self._write_single_example(feature_patch, writer, mode)
                        num_example += 1
                else:
                    self._write_single_example(feature, writer, mode)
                    num_example += 1
        return filename, num_example

    def _transform_one_slice(self, dictionary, index, mode):
        feature = self._get_dict_slice(dictionary, index)
        if self.ops_local:
            feature = self._preprocessing(feature, mode)
        return feature

    def _write_single_example(self, dictionary, writer, mode):
        feature_tfrecord = {}
        for key in self.feature_name[mode]:
            data = np.array(dictionary[key]).astype(self.feature_dtype[mode][key])
            expected_shape = self.feature_shape[mode][key]
            assert data.size > 0, "found empty data on feature '{}'".format(key)
            if len(expected_shape) > 1:
                assert expected_shape == data.shape, \
                    "inconsistent shape on same feature `{}` among different examples, expected `{}`, found `{}`" \
                    .format(key, expected_shape, data.shape)
            else:
                if data.size > 1:
                    assert max(data.shape) == np.prod(data.shape), "inconsistent shape on same feature `{}` among \
                        different examples, expected 0 or 1 dimensional array, found `{}`" \
                        .format(key, data.shape)
                    if not expected_shape:
                        self.feature_shape[mode][key] = [-1]
            feature_tfrecord[key] = self._bytes_feature(data.tostring())
        example = tf.train.Example(features=tf.train.Features(feature=feature_tfrecord))
        writer.write(example.SerializeToString())

    @staticmethod
    def _get_dict_slice(dictionary, index, keys=None):
        feature_slice = dict()
        if keys is None:
            keys = dictionary.keys()
        for key in keys:
            feature_slice[key] = dictionary[key][index]
        return feature_slice

    def _preprocessing(self, feature, mode):
        data = None
        for op in self.mode_ops[mode]:
            data = get_inputs_by_op(op, feature, data)
            data = op.forward(data, state={"mode": mode})
            if op.outputs:
                feature = write_outputs_by_key(feature, data, op.outputs)
        return feature

    def _verify_dict(self, dictionary, mode=None):
        if mode is None:
            feature_name = dictionary.keys()
        else:
            feature_name = self.feature_name[mode]
        num_example_list = []
        for key in feature_name:
            feature_data = dictionary[key]
            if isinstance(feature_data, list):
                num_example_list.append(len(feature_data))
            elif isinstance(feature_data, np.ndarray):
                num_example_list.append(feature_data.shape[0])
            else:
                raise ValueError("the feature only supports list or numpy array, unsupported key {}".format(key))
        assert len(set(num_example_list)) == 1, "features should have the same number of examples"
        return set(num_example_list).pop()

    def _prepare_train(self):
        if isinstance(self.train_data_local, str):
            df = pd.read_csv(self.train_data_local)
            self.train_data_local = df.to_dict('list')
        self.num_example_csv["train"] = self._verify_dict(self.train_data_local)
        self._check_ops("train")

    def _prepare_validation(self):
        if isinstance(self.validation_data_local, float):
            num_example_takeout = int(self.validation_data_local * self.num_example_csv["train"])
            train_idx = range(self.num_example_csv["train"])
            eval_idx = np.random.choice(train_idx, num_example_takeout, replace=False)
            train_idx = np.delete(train_idx, eval_idx)
            self.validation_data_local = {}
            for key in self.train_data_local.keys():
                total_data = self.train_data_local[key]
                if isinstance(total_data, list):
                    self.train_data_local[key] = [total_data[x] for x in train_idx]
                    self.validation_data_local[key] = [total_data[x] for x in eval_idx]
                else:
                    self.train_data_local[key] = total_data[train_idx]
                    self.validation_data_local[key] = total_data[eval_idx]
            self.num_example_csv["train"] = self.num_example_csv["train"] - num_example_takeout
        elif isinstance(self.validation_data_local, str):
            df = pd.read_csv(self.validation_data_local)
            self.validation_data_local = df.to_dict('list')
        self.num_example_csv["eval"] = self._verify_dict(self.validation_data_local)
        self._check_ops("eval")

    def _check_ops(self, mode):
        if self.ops_local:
            self.mode_ops[mode] = get_op_from_mode(self.ops_local, mode)
            if len(self.mode_ops[mode]) > 0:
                verify_ops(self.mode_ops[mode], "RecordWriter")

    def _prepare_savepath(self, save_dir):
        self.save_dir = save_dir
        if os.path.exists(self.save_dir):
            assert len(os.listdir(self.save_dir)) == 0, "Cannot save to {} because the directory is not empty".format(
                self.save_dir)
        else:
            os.makedirs(self.save_dir)
        print("FastEstimator: Saving tfrecord to %s" % self.save_dir)

    def _write_json_summary(self, mode):
        summary = {"feature_dtype": self.feature_dtype[mode], "feature_shape": self.feature_shape[mode]}
        files, num_examples = zip(*self.num_example_record[mode])
        summary["file_names"] = list(files)
        summary["num_examples"] = list(num_examples)
        summary["example_size_mb"] = self.mb_per_record_example[mode]
        if self.compression:
            summary["compression"] = self.compression
        file_name = "%s_summary%d.json" % (mode, self.feature_set_idx)
        with open(os.path.join(self.save_dir, file_name), 'w') as fp:
            json.dump(summary, fp, indent=4)
        self.global_file_idx[mode] += len(files)

    def transform(self, data, mode):
        assert isinstance(data, dict), "please provide dictionary with different features as key"
        assert len(self.ops) <= 1, "transform does not support unpaired dataset yet"
        num_data = self._verify_dict(dictionary=data)
        self.ops_local = self.ops[0]
        self._check_ops(mode)
        result = {}
        for idx in range(num_data):
            feature = self._transform_one_slice(data, idx, mode)
            if idx == 0:
                for key, value in feature.items():
                    result[key] = [value]
            else:
                for key, value in feature.items():
                    result[key].append(value)
        return result
