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

import numpy as np
import tensorflow as tf

from fastestimator.pipeline.processing import TensorFilter
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import get_op_from_mode, verify_ops
from fastestimator.util.tfrecord import get_features
from fastestimator.util.util import convert_tf_dtype
from fastestimator.util.schedule import Scheduler


class Pipeline:
    def __init__(self, batch_size, data=None, ops=None, read_feature=None, padded_batch=False, expand_dims=False,
                 max_shuffle_buffer_mb=3000):

        self.batch_size = batch_size
        self.data = data
        self.ops = ops
        self.read_feature = read_feature
        self.padded_batch = padded_batch
        self.expand_dims = expand_dims
        self.max_shuffle_buffer_mb = max_shuffle_buffer_mb
        self.possible_mode = ["train", "eval"]
        self.num_core = mp.cpu_count()
        self._verify_input()
        self._reset()

    def _verify_input(self):
        if self.data:
            assert isinstance(self.data, (dict, RecordWriter)), "data must be either RecordWriter, dictionary"
        if self.read_feature:
            assert isinstance(self.read_feature, (list, tuple, dict)), "write_feature must be either list, tuple or dictionary"
            if not isinstance(self.read_feature, tuple):
                self.read_feature = [self.read_feature]
        if not isinstance(self.ops, list):
            self.ops = [self.ops]

    def _reset(self):
        self.mode_list = []
        self.all_features = {"train": [], "eval": []}
        self.num_examples = {"train": [], "eval": []}
        self.shuffle_buffer = {"train": [], "eval": []}
        self.feature_name = {"train": [], "eval": []}
        self.extracted_dataset = {}
        self.transformed_dataset = {}
        #TFrecord only
        self.summary_file = {}
        self.feature_dtype = {"train": [], "eval": []}
        self.record_feature_shape = {"train": [], "eval": []}
        self.compression = {"train": [], "eval": []}
        self.file_names = {"train": [], "eval": []}

    def _prepare(self, inputs):
        self.inputs = inputs
        if isinstance(self.data, dict):
            self._get_numpy_config()
        else:
            assert inputs, "Must specify the data path of tfrecords"
            if isinstance(self.data, RecordWriter) and (not os.path.exists(inputs) or len(os.listdir(inputs)) == 0):
                self.data.create_tfrecord(save_dir=inputs)
            else:
                print("FastEstimator: Reading non-empty directory: {}".format(inputs))
            self._get_tfrecord_config()
        for mode in self.mode_list:
            self._get_feature_name(mode)
            self._extract_dataset(mode)
            self._transform_dataset(mode)

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
        assert len(set(num_examples_list)) == 1, "inconsistent number of data found during {}, please check the data".format(mode)
        self.num_examples[mode].append(set(num_examples_list).pop())
        self.shuffle_buffer[mode].append(set(num_examples_list).pop())
    
    def _get_tfrecord_config(self):
        found_data = False
        for mode in self.possible_mode:
            self.summary_file[mode] = [os.path.join(self.inputs, f) for f in os.listdir(self.inputs) if f.endswith(".json") and f.startswith("%s_summary" % mode)]
            if len(self.summary_file[mode]) > 0:
                self.mode_list.append(mode)
                self._get_tfrecord_config_mode(mode)
                found_data = True
        assert found_data, "could not find data summary file in {}".format(self.inputs)

    def _get_tfrecord_config_mode(self, mode):
        for json_file in self.summary_file[mode]:
            with open(json_file, 'r') as fp:
                summary = json.load(fp)
            file_names = [os.path.join(self.inputs, f) for f in summary["file_names"]]
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
            print("FastEstimator: Found %d examples for %s in %s" % (num_examples, mode, json_file))

    def _get_feature_name(self, mode):
        if len(self.all_features[mode]) > 1 and self.read_feature:
            assert isinstance(self.read_feature, tuple), "read feature must be a tuple for unpaired feature set"
            assert len(self.read_feature) == len(self.all_features[mode]), "the tuple should be consistent between read_feature and data"
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
            if isinstance(self.data, (list, tuple)):
                ds = tf.data.Dataset.from_tensor_slices(self.all_features[mode][idx])
            else:
                if mode == "train":
                    ds = tf.data.Dataset.from_tensor_slices(self.file_names[mode][idx])
                    ds = ds.shuffle(len(self.file_names[mode][idx]))
                    ds = ds.interleave(lambda ds: tf.data.TFRecordDataset(ds, compression_type=self.compression[mode][idx]), cycle_length=self.num_core, block_length=2)
                else:
                    ds = tf.data.TFRecordDataset(self.file_names[mode][idx], compression_type=self.compression[mode][idx])
                ds = ds.map(lambda ds: self._decode_records(ds, mode, idx), num_parallel_calls=self.num_core)
            ds = ds.shuffle(self.shuffle_buffer[mode][idx])
            ds_tuple += ds,
        # Combine dataset from different unpaired feature sets
        if len(self.all_features[mode]) > 1:
            dataset = tf.data.Dataset.zip(ds_tuple)
            dataset = dataset.map(self._combine_dataset, num_parallel_calls=self.num_core)
        else:
            dataset = ds_tuple[0]
        self.extracted_dataset[mode] = dataset

    def _transform_dataset(self, mode):
        dataset = self.extracted_dataset[mode]
        signature_epoch_list = [0]
        if self.ops:
            pass
        
        for epoch in signature_epoch_list:
            





    def _epoch_dataset(self, state):
        epoch = state["epoch"]
        mode = state["mode"]
        



    def _execute_mode_ops(self, dataset, state):
        num_filters = len(self.mode_filter_ops[mode])
        ops = self.mode_forward_ops[mode][0]
        dataset = dataset.map(lambda dataset: self._preprocess_fn(dataset, ops), num_parallel_calls=self.num_core)
        if num_filters > 0:
            for idx in range(num_filters):
                dataset = dataset.filter(
                    lambda dataset: self.mode_filter_ops[mode][idx].forward(dataset, {"mode": mode}))
                ops = self.mode_forward_ops[mode][idx + 1]
                if len(ops) > 0:
                    dataset = dataset.map(lambda dataset: self._preprocess_fn(dataset, ops),
                                          num_parallel_calls=self.num_core)
        return dataset

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

    def _combine_dataset(self, *dataset):
        combined_dict = {}
        for ds in dataset:
            for key in ds.keys():
                assert key not in combined_dict, "found duplicated key {} in unpaird feature sets".format(key)
                combined_dict[key] = ds[key]
        return combined_dict

    def _preprocess_fn(self, feature, ops):
        data = None
        for op in ops:
            if op.inputs:
                data = self._get_inputs_from_key(feature, op.inputs)
            data = op.forward(data, {})
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


    def show_results(self, inputs=None, mode="train", num_steps=1, current_epoch=0):
        data = []
        self._prepare(inputs=inputs)
        state = {"mode":mode, "epoch": current_epoch}
        dataset = self._input_epoch(state)
        for example in dataset.take(num_steps):
            data.append(example)
        self._reset()
        return data

    def benchmark(self, inputs=None, mode="train", num_steps=1000, log_interval=100, current_epoch=0):
        self._prepare(inputs=inputs)
        state = {"mode":mode, "epoch": current_epoch}
        dataset = self._input_epoch(state)
        start = time.time()
        for _ in dataset.take(num_steps):
            if step.numpy() % log_interval == 0 and step.numpy() > 0:
                duration = time.time() - start
                example_per_sec = log_interval * self.batch_size_tensor.numpy() / duration
                tf.print("FastEstimator: Step: %d, Epoch: %d, Batch Size %d, Example/sec %.2f" % (step.numpy(), epoch.numpy(), self.batch_size_tensor.numpy(), example_per_sec))
                start = time.time()
        self._reset()
