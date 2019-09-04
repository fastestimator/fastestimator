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
import json
import multiprocessing as mp
import os
import time
from itertools import chain

import numpy as np
import tensorflow as tf

from fastestimator.pipeline.processing import TensorFilter
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import (get_inputs_by_key,
                                   get_inputs_by_op,
                                   get_op_from_mode,
                                   verify_ops,
                                   write_outputs_by_key)
from fastestimator.util.schedule import Scheduler
from fastestimator.util.tfrecord import get_features
from fastestimator.util.util import convert_tf_dtype, get_num_devices


class Pipeline:
    def __init__(self,
                 data,
                 batch_size,
                 ops=None,
                 read_feature=None,
                 padded_batch=False,
                 expand_dims=False,
                 max_shuffle_buffer_mb=3000):

        self.batch_size = batch_size
        self.data = data
        self.ops = ops
        self.read_feature = read_feature
        self.padded_batch = padded_batch
        self.expand_dims = expand_dims
        self.max_shuffle_buffer_mb = max_shuffle_buffer_mb
        self.possible_mode = ["train", "eval"]
        self.padded_shape = None
        self.global_batch_multiplier = 1
        self.num_core = mp.cpu_count()
        self._verify_input()
        self._reset()

    def get_all_output_keys(self):
        return set(
            chain.from_iterable(
                map(
                    lambda x: [x] if isinstance(x, str) or x is None else x,
                    map(
                        lambda x: x.outputs,
                        list(
                            chain.from_iterable([
                                list(op.epoch_dict.values()) if isinstance(op, Scheduler) else [op] for op in self.ops
                            ])))))) - {None} | set(
                                chain.from_iterable(list(chain.from_iterable(self.feature_name.values()))))

    def _verify_input(self):
        assert isinstance(self.data, (dict, RecordWriter, str)), \
            "data must be either RecordWriter, dictionary or record path"
        if self.read_feature:
            assert isinstance(self.read_feature,
                              (list, tuple, dict)), "read_feature must be either list, tuple or dictionary"
            if not isinstance(self.read_feature, tuple):
                self.read_feature = [self.read_feature]
        if self.ops:
            if not isinstance(self.ops, list):
                self.ops = [self.ops]
        else:
            self.ops = []

    def _reset(self):
        self.mode_list = []
        self.all_features = {"train": [], "eval": []}
        self.num_examples = {"train": [], "eval": []}
        self.shuffle_buffer = {"train": [], "eval": []}
        self.feature_name = {"train": [], "eval": []}
        self.extracted_dataset = {}
        self.transformed_dataset = {}
        self.dataset_schedule = {}
        # TFrecord only
        self.summary_file = {}
        self.feature_dtype = {"train": [], "eval": []}
        self.record_feature_shape = {"train": [], "eval": []}
        self.compression = {"train": [], "eval": []}
        self.file_names = {"train": [], "eval": []}
        self.global_batch_multiplier = 1

    def prepare(self, distribute_strategy=None):
        if isinstance(self.data, dict):
            self._get_numpy_config()
        elif isinstance(self.data, (RecordWriter, str)):
            if isinstance(self.data, RecordWriter):
                data_path = self.data.save_dir
                if not os.path.exists(data_path) or not os.listdir(data_path):
                    self.data.write()
            else:
                data_path = self.data
            print("FastEstimator: Reading non-empty directory: {}".format(data_path))
            self._get_tfrecord_config(data_path)
        else:
            raise ValueError("data must be one of the following: dictionary, RecordWriter or record path")
        for mode in self.mode_list:
            self._get_feature_name(mode)
            self._extract_dataset(mode)
            self._transform_dataset(mode, distribute_strategy)

    def _get_numpy_config(self):
        for mode in self.possible_mode:
            if mode in self.data:
                self.mode_list.append(mode)
                self.all_features[mode].append(self.data[mode])
                self._get_numpy_config_mode(mode)

    def _get_numpy_config_mode(self, mode):
        num_examples_list = []
        data = self.data[mode]
        for key in data.keys():
            feature_data = data[key]
            if type(feature_data) is list:
                num_examples_list.append(len(feature_data))
            elif type(feature_data) is np.ndarray:
                num_examples_list.append(feature_data.shape[0])
            else:
                raise ValueError("the feature only supports list or numpy array")
        assert len(set(
            num_examples_list)) == 1, "inconsistent number of data found during {}, please check the data".format(mode)
        self.num_examples[mode].append(set(num_examples_list).pop())
        self.shuffle_buffer[mode].append(set(num_examples_list).pop())

    def _get_tfrecord_config(self, data_path):
        found_data = False
        for mode in self.possible_mode:
            self.summary_file[mode] = [
                os.path.join(data_path, f) for f in os.listdir(data_path)
                if f.endswith(".json") and f.startswith("%s_summary" % mode)
            ]
            if len(self.summary_file[mode]) > 0:
                self.mode_list.append(mode)
                self._get_tfrecord_config_mode(data_path, mode)
                found_data = True
        assert found_data, "could not find data summary file in {}".format(data_path)

    def _get_tfrecord_config_mode(self, data_path, mode):
        for json_file in self.summary_file[mode]:
            with open(json_file, 'r') as fp:
                summary = json.load(fp)
            file_names = [os.path.join(data_path, f) for f in summary["file_names"]]
            self.file_names[mode].append(file_names)
            num_examples = np.sum(summary["num_examples"])
            example_size_mb = summary["example_size_mb"]
            self.num_examples[mode].append(num_examples)
            self.feature_dtype[mode].append(summary["feature_dtype"])
            self.record_feature_shape[mode].append(summary["feature_shape"])
            if "compression" in summary:
                compression = summary["compression"]
            else:
                compression = None
            self.compression[mode].append(compression)
            self.all_features[mode].append(get_features(file_names[0], compression=compression))
            self.shuffle_buffer[mode].append(int(min(num_examples, self.max_shuffle_buffer_mb // example_size_mb)))
            print("FastEstimator: Found %d examples for %s in %s" % (int(num_examples), mode, json_file))

    def _get_feature_name(self, mode):
        if len(self.all_features[mode]) > 1 and self.read_feature:
            assert isinstance(self.read_feature, tuple), "read feature must be a tuple for unpaired feature set"
            assert len(self.read_feature) == len(
                self.all_features[mode]), "the tuple should be consistent between read_feature and data"
        for idx, feature in enumerate(self.all_features[mode]):
            if self.read_feature is None:
                self.feature_name[mode].append(list(feature.keys()))
            elif isinstance(self.read_feature[idx], dict):
                self.feature_name[mode].append(self.read_feature[idx][mode])
            else:
                self.feature_name[mode].append(self.read_feature[idx])

    def _extract_dataset(self, mode):
        ds_tuple = ()
        # Data Reading
        for idx in range(len(self.all_features[mode])):
            if isinstance(self.data, dict):
                ds = tf.data.Dataset.from_tensor_slices(self.all_features[mode][idx])
            else:
                if mode == "train":
                    ds = tf.data.Dataset.from_tensor_slices(self.file_names[mode][idx])
                    ds = ds.shuffle(len(self.file_names[mode][idx]))
                    ds = ds.interleave(
                        lambda ds_lam: tf.data.TFRecordDataset(ds_lam, compression_type=self.compression[mode][idx]),
                        cycle_length=self.num_core,
                        block_length=2)
                else:
                    ds = tf.data.TFRecordDataset(self.file_names[mode][idx],
                                                 compression_type=self.compression[mode][idx])
                ds = ds.map(lambda ds_lam: self._decode_records(ds_lam, mode, idx), num_parallel_calls=self.num_core)
            ds = ds.shuffle(self.shuffle_buffer[mode][idx])
            ds = ds.repeat()
            ds_tuple += ds,
        # Combine dataset from different unpaired feature sets
        if len(self.all_features[mode]) > 1:
            dataset = tf.data.Dataset.zip(ds_tuple)
            dataset = dataset.map(self._combine_dataset, num_parallel_calls=self.num_core)
        else:
            dataset = ds_tuple[0]
        self.extracted_dataset[mode] = dataset

    def _transform_dataset(self, mode, distribute_strategy):
        signature_epoch, mode_ops = self._get_signature_epoch(mode)
        extracted_ds = self.extracted_dataset[mode]
        state = {"mode": mode}
        dataset_map = {}
        for epoch in signature_epoch:
            epoch_ops_all = []
            forward_ops_epoch = []
            filter_ops_epoch = []
            forward_ops_between_filter = []
            # get batch size for the epoch
            global_batch_size = self.get_global_batch_size(epoch)
            # generate ops for specific mode and epoch
            for op in mode_ops:
                if isinstance(op, Scheduler):
                    scheduled_op = op.get_current_value(epoch)
                    if scheduled_op:
                        epoch_ops_all.append(scheduled_op)
                else:
                    epoch_ops_all.append(op)
            # check the ops
            epoch_ops_without_filter = [op for op in epoch_ops_all if not isinstance(op, TensorFilter)]
            verify_ops(epoch_ops_without_filter, "Pipeline")
            # arrange operation according to filter location
            for op in epoch_ops_all:
                if not isinstance(op, TensorFilter):
                    forward_ops_between_filter.append(op)
                else:
                    forward_ops_epoch.append(forward_ops_between_filter)
                    filter_ops_epoch.append(op)
                    forward_ops_between_filter = []
            forward_ops_epoch.append(forward_ops_between_filter)
            # execute the operations
            dataset = self._execute_ops(extracted_ds, forward_ops_epoch, filter_ops_epoch, state)
            # rest of the dataset setup
            if self.expand_dims:
                dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
            if self.padded_batch:
                _ = dataset.map(self._get_padded_shape)
                dataset = dataset.padded_batch(global_batch_size, padded_shapes=self.padded_shape)
            else:
                dataset = dataset.batch(global_batch_size)
            dataset = dataset.prefetch(buffer_size=1)
            if distribute_strategy:
                dataset = distribute_strategy.experimental_distribute_dataset(dataset)
            dataset_map[epoch] = iter(dataset)
        self.dataset_schedule[mode] = Scheduler(epoch_dict=dataset_map)

    def _get_padded_shape(self, dataset):
        padded_shape = {}
        for key in dataset:
            padded_shape[key] = dataset[key].shape
        self.padded_shape = padded_shape
        return dataset

    def _execute_ops(self, dataset, forward_ops_epoch, filter_ops_epoch, state):
        num_filters = len(filter_ops_epoch)
        forward_ops = forward_ops_epoch[0]
        dataset = dataset.map(lambda ds: self._preprocess_fn(ds, forward_ops, state), num_parallel_calls=self.num_core)
        if num_filters > 0:
            for filter_op, forward_ops in zip(filter_ops_epoch, forward_ops_epoch[1:]):
                dataset = dataset.filter(lambda ds: self._filter_fn(ds, filter_op, state))
                dataset = dataset.map(lambda ds: self._preprocess_fn(ds, forward_ops, state),
                                      num_parallel_calls=self.num_core)
        return dataset

    @staticmethod
    def _filter_fn(feature, filter_op, state):
        data = get_inputs_by_key(feature, filter_op.inputs)
        data = filter_op.forward(data, state)
        return data

    @staticmethod
    def _preprocess_fn(feature, forward_ops, state):
        data = None
        for op in forward_ops:
            data = get_inputs_by_op(op, feature, data)
            data = op.forward(data, state)
            if op.outputs:
                feature = write_outputs_by_key(feature, data, op.outputs)
        return feature

    def _get_signature_epoch(self, mode):
        signature_epoch = [0]
        if isinstance(self.batch_size, Scheduler):
            signature_epoch.extend(self.batch_size.keys)
        mode_ops = get_op_from_mode(self.ops, mode)
        for op in mode_ops:
            if isinstance(op, Scheduler):
                signature_epoch.extend(op.keys)
        return list(set(signature_epoch)), mode_ops

    def _decode_records(self, dataset, mode, idx):
        decoded_data = {}
        all_data = tf.io.parse_single_example(dataset, features=self.all_features[mode][idx])
        for feature in self.feature_name[mode][idx]:
            data = all_data[feature]
            if "str" in str(data.dtype) and "str" not in self.feature_dtype[mode][idx][feature]:
                data = tf.io.decode_raw(data, convert_tf_dtype(self.feature_dtype[mode][idx][feature]))
                data = tf.reshape(data, self.record_feature_shape[mode][idx][feature])
            if "int" in str(data.dtype):
                data = tf.cast(data, tf.int32)
            elif "str" not in self.feature_dtype[mode][idx][feature]:
                data = tf.cast(data, tf.float32)
            decoded_data[feature] = data
        return decoded_data

    @staticmethod
    def _combine_dataset(*dataset):
        combined_dict = {}
        for ds in dataset:
            for key in ds.keys():
                assert key not in combined_dict, "found duplicated key {} in unpaird feature sets".format(key)
                combined_dict[key] = ds[key]
        return combined_dict

    def get_global_batch_size(self, epoch):
        batch_per_device = self.batch_size
        if isinstance(batch_per_device, Scheduler):
            batch_per_device = batch_per_device.get_current_value(epoch)
        global_batch_size = batch_per_device * self.global_batch_multiplier
        return global_batch_size

    def show_results(self, mode="train", num_steps=1, current_epoch=0):
        data = []
        self.global_batch_multiplier = get_num_devices()
        self.prepare()
        ds_iter = self.dataset_schedule[mode].get_current_value(current_epoch)
        for _ in range(num_steps):
            data.append(next(ds_iter))
        self._reset()
        return data

    def benchmark(self, mode="train", num_steps=1000, log_interval=100, current_epoch=0):
        self.global_batch_multiplier = get_num_devices()
        global_batch_size = self.get_global_batch_size(current_epoch)
        self.prepare()
        ds_iter = self.dataset_schedule[mode].get_current_value(current_epoch)
        start = time.perf_counter()
        for idx in range(num_steps + 1):
            _ = next(ds_iter)
            if idx % log_interval == 0:
                if idx == 0:
                    start = time.perf_counter()
                else:
                    duration = time.perf_counter() - start
                    example_per_sec = log_interval * global_batch_size / duration
                    print("FastEstimator: Step: %d, Epoch: %d, Batch Size %d, Example/sec %.2f" %
                          (idx, current_epoch, global_batch_size, example_per_sec))
                    start = time.perf_counter()
        self._reset()
