# Copyright 2026 The FastEstimator Authors. All Rights Reserved.
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
import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate.elastic_transform_3d import ElasticTransform3D
from fastestimator.op.numpyop.multivariate.random_affine_3d import RandomAffine3D
from fastestimator.op.numpyop.multivariate.random_crop_3d import RandomCrop3D
from fastestimator.op.numpyop.multivariate.random_flip_3d import RandomFlip3D
from fastestimator.op.numpyop.multivariate.random_rotate_3d import RandomRotate3D
from fastestimator.op.numpyop.univariate.gaussian_blur_3d import GaussianBlur3D
from fastestimator.op.numpyop.univariate.gaussian_noise_3d import GaussianNoise3D


class TestMedical3DPipeline(unittest.TestCase):
    """Integration test that chains multiple 3D augmentations together to simulate a real medical imaging pipeline."""
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.volume = np.random.rand(32, 64, 64).astype(np.float32)
        cls.mask = (np.random.rand(32, 64, 64) > 0.5).astype(np.float32)

    def test_chained_augmentation_pipeline(self):
        """Run volume through a sequence of augmentations and verify shapes are preserved."""
        data = [self.volume.copy(), self.mask.copy()]

        # Step 1: Random flip
        flip = RandomFlip3D(inputs=('x', 'mask'), outputs=('x', 'mask'))
        data = flip.forward(data=data, state={})
        self.assertEqual(data[0].shape, (32, 64, 64))
        self.assertEqual(data[1].shape, (32, 64, 64))

        # Step 2: Random crop
        crop = RandomCrop3D(inputs=('x', 'mask'), outputs=('x', 'mask'), crop_size=(16, 32, 32))
        data = crop.forward(data=data, state={})
        self.assertEqual(data[0].shape, (16, 32, 32))
        self.assertEqual(data[1].shape, (16, 32, 32))

        # Step 3: Add noise to image only
        noise = GaussianNoise3D(inputs='x', outputs='x', std_range=(0.01, 0.05))
        noised = noise.forward(data=[data[0]], state={})
        data[0] = noised[0]
        self.assertEqual(data[0].shape, (16, 32, 32))

        # Step 4: Blur image only
        blur = GaussianBlur3D(inputs='x', outputs='x', sigma_range=(0.5, 1.0))
        blurred = blur.forward(data=[data[0]], state={})
        data[0] = blurred[0]
        self.assertEqual(data[0].shape, (16, 32, 32))

    def test_elastic_with_mask_consistency(self):
        """Elastic transform applied to image and mask should produce same-shape outputs."""
        elastic = ElasticTransform3D(inputs=('x', 'mask'),
                                     outputs=('x', 'mask'),
                                     alpha=100,
                                     sigma=10,
                                     label_keys='mask')
        data = elastic.forward(data=[self.volume.copy(), self.mask.copy()], state={})
        self.assertEqual(data[0].shape, self.volume.shape)
        self.assertEqual(data[1].shape, self.mask.shape)
        self.assertEqual(data[0].dtype, np.float32)
        self.assertEqual(data[1].dtype, np.float32)

    def test_affine_with_mask_consistency(self):
        """Affine transform applied to image and mask should produce same-shape outputs."""
        affine = RandomAffine3D(inputs=('x', 'mask'),
                                outputs=('x', 'mask'),
                                rotate_range=15.0,
                                scale_range=0.1,
                                translate_range=5.0,
                                label_keys='mask')
        data = affine.forward(data=[self.volume.copy(), self.mask.copy()], state={})
        self.assertEqual(data[0].shape, self.volume.shape)
        self.assertEqual(data[1].shape, self.mask.shape)
        self.assertEqual(data[0].dtype, np.float32)
        self.assertEqual(data[1].dtype, np.float32)

    def test_rotate_with_label_keys(self):
        """Rotation with label_keys should use nearest-neighbor interpolation for masks."""
        rotate = RandomRotate3D(inputs=('x', 'mask'), outputs=('x', 'mask'), angle_range=15.0, label_keys='mask')
        data = rotate.forward(data=[self.volume.copy(), self.mask.copy()], state={})
        self.assertEqual(data[0].shape, self.volume.shape)
        self.assertEqual(data[1].shape, self.mask.shape)

    def test_4d_volume_pipeline(self):
        """Test the pipeline with multi-channel (D, H, W, C) volumes."""
        vol_4d = np.random.rand(16, 32, 32, 2).astype(np.float32)

        flip = RandomFlip3D(inputs='x', outputs='x')
        data = flip.forward(data=[vol_4d.copy()], state={})
        self.assertEqual(data[0].shape, (16, 32, 32, 2))

        noise = GaussianNoise3D(inputs='x', outputs='x')
        data = noise.forward(data=data, state={})
        self.assertEqual(data[0].shape, (16, 32, 32, 2))

        blur = GaussianBlur3D(inputs='x', outputs='x')
        data = blur.forward(data=data, state={})
        self.assertEqual(data[0].shape, (16, 32, 32, 2))

        crop = RandomCrop3D(inputs='x', outputs='x', crop_size=(8, 16, 16))
        data = crop.forward(data=data, state={})
        self.assertEqual(data[0].shape, (8, 16, 16, 2))
