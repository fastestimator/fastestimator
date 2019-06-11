from unittest import TestCase
from tensorflow.python.client import device_lib
from .util import get_num_GPU


class TestUtil(TestCase):
    def test_get_num_GPU(self):
        local_device_protos = device_lib.list_local_devices()
        num_gpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])

        assert num_gpu == get_num_GPU()
