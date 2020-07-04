import csv
import os
import shutil
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal, sample_system_object, sample_system_object_torch
from fastestimator.trace.io import TensorBoard
from fastestimator.trace.io.tensorboard import _TfWriter, _TorchWriter
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import FeInputSpec


def getfilepath():
    path = os.path.join(*[tempfile.gettempdir(), 'tensorboard', 'train'])
    for filename in os.listdir(path):
        return os.path.join(path, filename)


class TestTensorboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = Data({
            'x': tf.random.normal(shape=(1, 28, 28, 3)),
            'y': tf.random.uniform(shape=(1, ), maxval=10, dtype=tf.int32),
            'images': tf.random.normal(shape=(1, 28, 28, 3)),
            'embed': np.ones(shape=(1, 3, 3, 3)),
            'embed_images': np.ones(shape=(1, 3, 3, 3))
        })
        cls.torch_data = Data({
            'x': torch.rand(size=(1, 1, 28, 28)),
            'y': torch.rand(size=(3, )),
            'images': torch.rand(size=(1, 3, 28, 28)),
            'embed': np.ones(shape=(1, 3, 3, 3)),
            'embed_images': np.ones(shape=(1, 3, 3, 3))
        })
        cls.log_dir = os.path.join(tempfile.gettempdir(), 'tensorboard')
        cls.train_path = os.path.join(cls.log_dir, 'train')
        cls.embed_path = os.path.join(cls.log_dir, 'train', '00001', 'embed')
        cls.on_begin_msg = "FastEstimator-Tensorboard: writing logs to {}".format(cls.log_dir)

    def test_tf_on_begin(self):
        tensorboard = TensorBoard(log_dir=self.log_dir)
        tensorboard.system = sample_system_object()
        tensorboard.system.global_step = 1
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            tensorboard.on_begin(data=self.tf_data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, self.on_begin_msg)

    def test_tf_on_batch_end(self):
        tensorboard = TensorBoard(log_dir=self.log_dir, weight_histogram_freq=1, update_freq=1)
        tensorboard.system = sample_system_object()
        tensorboard.system.global_step = 1
        tensorboard.writer = _TfWriter(self.log_dir, '', tensorboard.system.network)
        model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam')
        tensorboard.system.network.epoch_models = {model}
        if os.path.exists(self.train_path):
            shutil.rmtree(self.train_path)
        tensorboard.on_batch_end(data=self.tf_data)
        filepath = getfilepath()
        for e in tf.compat.v1.train.summary_iterator(filepath):
            for v in e.summary.value:
                if v.tag == "tf_dense_1/bias_0":
                    output = v.histo.num
                    self.assertEqual(output, 10.0)

    def test_tf_on_epoch_end(self):
        tensorboard = TensorBoard(log_dir=self.log_dir,
                                  weight_histogram_freq=1,
                                  update_freq=1,
                                  write_images='images',
                                  write_embeddings='embed',
                                  embedding_images='embed_images')
        tensorboard.system = sample_system_object()
        tensorboard.system.global_step = 1
        tensorboard.writer = _TfWriter(self.log_dir, '', tensorboard.system.network)
        model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam')
        tensorboard.system.network.epoch_models = {model}
        if os.path.exists(self.train_path):
            shutil.rmtree(self.train_path)
        tensorboard.on_epoch_end(data=self.tf_data)
        tsv_path = os.path.join(self.embed_path, 'tensors.tsv')
        embed_img_path = os.path.join(self.embed_path, 'sprite.png')
        # get tensor data from tsv file
        fo = open(tsv_path)
        tsv_content = csv.reader(fo, delimiter='\t')
        for row in tsv_content:
            tsv_data = row
        fo.close()
        # get the image data
        output_img = np.asarray(Image.open(embed_img_path))
        with self.subTest('Check if tensors.tsv was generated'):
            self.assertTrue(os.path.exists(tsv_path))
        with self.subTest('Check if embed image was generated'):
            self.assertTrue(os.path.exists(embed_img_path))
        with self.subTest('Check content of tensors.tsv'):
            self.assertEqual(tsv_data, 27 * ['1.0'])
        with self.subTest('Check embed image content'):
            self.assertTrue(is_equal(output_img, 255 * np.ones(shape=(3, 3, 3), dtype=np.int)))

    def test_torch_on_begin(self):
        tensorboard = TensorBoard(log_dir=self.log_dir)
        tensorboard.system = sample_system_object_torch()
        tensorboard.system.global_step = 1
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            tensorboard.on_begin(data=self.torch_data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, self.on_begin_msg)

    def test_torch_on_batch_end(self):
        tensorboard = TensorBoard(log_dir=self.log_dir, weight_histogram_freq=1, update_freq=1)
        tensorboard.system = sample_system_object_torch()
        tensorboard.system.global_step = 1
        tensorboard.writer = _TorchWriter(self.log_dir, '', tensorboard.system.network)
        model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
        model.fe_input_spec = FeInputSpec(self.torch_data['x'], model)
        tensorboard.system.network.epoch_models = {model}
        if os.path.exists(self.train_path):
            shutil.rmtree(self.train_path)
        tensorboard.on_batch_end(data=self.torch_data)
        filepath = getfilepath()
        for e in tf.compat.v1.train.summary_iterator(filepath):
            for v in e.summary.value:
                if v.tag == "torch_fc1/bias":
                    output = v.histo.num
                    self.assertEqual(output, 64.0)

    def test_torch_on_epoch_end(self):
        tensorboard = TensorBoard(log_dir=self.log_dir,
                                  weight_histogram_freq=1,
                                  update_freq=1,
                                  write_images='images',
                                  write_embeddings='embed',
                                  embedding_images='embed_images')
        tensorboard.system = sample_system_object_torch()
        tensorboard.system.global_step = 1
        tensorboard.writer = _TorchWriter(self.log_dir, '', tensorboard.system.network)
        model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam')
        tensorboard.system.network.epoch_models = {model}
        if os.path.exists(self.train_path):
            shutil.rmtree(self.train_path)
        tensorboard.on_epoch_end(data=self.torch_data)
        tsv_path = os.path.join(self.embed_path, 'tensors.tsv')
        embed_img_path = os.path.join(self.embed_path, 'sprite.png')
        # get tensor data from tsv file
        fo = open(tsv_path)
        tsv_content = csv.reader(fo, delimiter='\t')
        for row in tsv_content:
            tsv_data = row
        fo.close()
        # get the image data
        output_img = np.asarray(Image.open(embed_img_path))
        with self.subTest('Check if tensors.tsv was generated'):
            self.assertTrue(os.path.exists(tsv_path))
        with self.subTest('Check if embed image was generated'):
            self.assertTrue(os.path.exists(embed_img_path))
        with self.subTest('Check content of tensors.tsv'):
            self.assertEqual(tsv_data, 27 * ['1.0'])
        with self.subTest('Check embed image content'):
            self.assertTrue(is_equal(output_img, 255 * np.ones(shape=(3, 3, 3), dtype=np.int)))
