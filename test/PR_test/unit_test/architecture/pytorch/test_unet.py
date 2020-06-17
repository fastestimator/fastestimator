import unittest

import numpy as np
import torch

from fastestimator.architecture.pytorch import UNet
from fastestimator.architecture.pytorch.unet import UNetDecoderBlock, UNetEncoderBlock


class TestUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_block_data = torch.Tensor(np.ones((128, 1, 3, 3)))
        cls.test_architecture_data = torch.Tensor(np.ones((1, 1, 128, 128)))

    def test_unet_encoder_block(self):
        unet_encoder_block = UNetEncoderBlock(in_channels=1, out_channels=128)
        output_shape = unet_encoder_block(self.test_block_data)[0].shape
        self.assertEqual(output_shape, (128, 128, 3, 3))

    def test_unet_decoder_block(self):
        unet_decoder_block = UNetDecoderBlock(in_channels=1, mid_channels=64, out_channels=128)
        output_shape = unet_decoder_block(self.test_block_data)[0].shape
        self.assertEqual(output_shape, (128, 6, 6))

    def test_unet(self):
        unet = UNet()
        output_shape = unet(self.test_architecture_data).detach().numpy().shape
        self.assertEqual(output_shape, (1, 1, 128, 128))
