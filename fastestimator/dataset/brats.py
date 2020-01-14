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
import multiprocessing as mp
import os
import shutil
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from random import shuffle

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from nilearn.image import reorder_img
from nilearn.image.image import _crop_img_to

labels = [1, 2, 4]  # segmentation ground truth has 3 different class
nlabels = len(labels)  # segmentation ground truth classes are 1 2 4
modalities = ["t1", "t1ce", "flair", "t2", "seg"]
no_bias_correction_modalities = ["flair", "seg"]


def _resize_nifti_images(data_nifti, resized_img_shape=(144, 144, 144), interpolation='linear'):
    data_nifti = reorder_img(data_nifti, resample=interpolation)  # bring pixel data to RAS format
    new_spacing = np.divide(np.multiply(data_nifti.header.get_data_shape(), data_nifti.header.get_zooms()),
                            resized_img_shape)

    data = data_nifti.get_data()
    data = np.rot90(data, 1, axes=(0, 2))  # converting to channel * width * height for sitk
    image_sitk = sitk.GetImageFromArray(data)
    image_sitk.SetSpacing(np.asarray(data_nifti.header.get_zooms(), dtype=np.float))
    new_size = np.asarray(resized_img_shape, dtype=np.int16)

    ref_image = sitk.GetImageFromArray(np.ones(new_size, dtype=np.float).T * 0.)
    ref_image.SetSpacing(new_spacing)
    ref_image.SetDirection(image_sitk.GetDirection())
    ref_image.SetOrigin(image_sitk.GetOrigin())

    resample_filter = sitk.ResampleImageFilter()
    transform = sitk.Transform()
    transform.SetIdentity()
    output_pixel_type = image_sitk.GetPixelID()
    resample_filter.SetOutputPixelType(output_pixel_type)
    if interpolation == 'nearest':
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)  # nearest is used for modalities t1,t2,flair,t1ce
    else:
        resample_filter.SetInterpolator(sitk.sitkLinear)  # linear intrepolation for segmentation mask
    resample_filter.SetTransform(transform)
    resample_filter.SetDefaultPixelValue(0.)
    resample_filter.SetReferenceImage(ref_image)
    resampled_image = resample_filter.Execute(image_sitk)
    data = sitk.GetArrayFromImage(resampled_image)
    data = np.rot90(data, -1, axes=(0, 2))  # converting to height * width * channel

    return data, new_spacing


def _generate_preprocessed_samples_core(path_brats_preprocessed,
                                        sample,
                                        mean_samples,
                                        std_smaples,
                                        resized_img_shape=(144, 144, 144)):

    modality_images = []
    modality_filenames = []
    for mod in modalities[:-1]:
        mod = '*' + mod + '.nii.gz'
        mod_wise_files = glob(os.path.join(sample, mod))
        assert len(mod_wise_files) == 1, 'Minimum one modality nifti image expected'
        modality_filenames.append(mod_wise_files[0])

    slices = _crop_to_non_zero_content(modality_filenames)

    for mod_fn in modality_filenames:
        data = nib.load(mod_fn)
        data = _crop_img_to(data, slices, copy=True)
        data, new_spacing_mi = _resize_nifti_images(data, interpolation='linear', resized_img_shape=resized_img_shape)
        modality_images.append(data)
    modality_images = np.asarray(modality_images, dtype=np.float32)
    modality_images -= mean_samples[:, np.newaxis, np.newaxis, np.newaxis]
    modality_images /= std_smaples[:, np.newaxis, np.newaxis, np.newaxis]

    affine = np.eye(4)
    new_spacing_mi = np.append(new_spacing_mi, 1)
    np.fill_diagonal(a=affine, val=new_spacing_mi)
    mod_data_resized_nifti = nib.Nifti1Image(modality_images, affine=affine)

    mod_filename = os.path.basename(sample) + '_mod.nii.gz'
    nib.save(mod_data_resized_nifti, os.path.join(path_brats_preprocessed, mod_filename))

    seg = '*' + 'seg' + '.nii.gz'
    seg_file = glob(os.path.join(sample, seg))
    assert len(seg_file) == 1, 'Minimum one segmeantion image expected'
    seg_data = nib.load(seg_file[0])
    seg_data = _crop_img_to(seg_data, slices, copy=True)
    seg_data, new_spacing_seg = _resize_nifti_images(seg_data, interpolation='nearest', resized_img_shape=resized_img_shape)
    seg_data = seg_data[np.newaxis]
    affine = np.eye(4)
    new_spacing_seg = np.append(new_spacing_seg, 1)
    np.fill_diagonal(a=affine, val=new_spacing_seg)

    seg_data_resized_nifti = nib.Nifti1Image(seg_data, affine=affine)
    seg_filename = os.path.basename(sample) + '_seg.nii.gz'
    nib.save(seg_data_resized_nifti, os.path.join(path_brats_preprocessed, seg_filename))


def _create_csv(path_brats, mode, samples):

    mod_data = []  # modality image data
    seg_mask_data = []  # segmentation mask

    for sample in samples:
        mod_filename = os.path.basename(sample) + '_mod.nii.gz'
        seg_filename = os.path.basename(sample) + '_seg.nii.gz'

        mod_data.append(glob(os.path.join(path_brats, 'preprocessed', mod_filename))[0])
        seg_mask_data.append(glob(os.path.join(path_brats, 'preprocessed', seg_filename))[0])

    df = pd.DataFrame({'mod_img': mod_data, 'seg_mask': seg_mask_data})
    csv_file_name = mode + '_brats' + '.csv'
    csv_file_name = os.path.join(path_brats, csv_file_name)
    df.to_csv(csv_file_name, index=False)
    return csv_file_name


def _generate_bias_corrected(sample):
    mod_files = glob(os.path.join(sample, '*.nii.gz'))
    for mod_fn in mod_files:
        if any([mod in mod_fn for mod in no_bias_correction_modalities]):
            dir_name = sample.replace('data', 'bias_corrected')
            file_name = os.path.basename(mod_fn)
            dest_file = os.path.join(dir_name, file_name)
            shutil.copy(mod_fn, dest_file)
        else:
            input_image = sitk.ReadImage(mod_fn, sitk.sitkFloat64)
            output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
            file_name = os.path.basename(mod_fn)
            dir_name = sample.replace('data', 'bias_corrected')
            os.makedirs(dir_name, exist_ok=True)
            output_file = os.path.join(dir_name, file_name)
            sitk.WriteImage(output_image, output_file)


def _generate_mean_std_from_sample(sample, resized_img_shape):
    modality_images = []
    modality_filenames = []
    for mod in modalities[:-1]:
        mod = '*' + mod + '.nii.gz'
        mod_wise_files = glob(os.path.join(sample, mod))
        assert len(mod_wise_files) == 1, 'Minimum one modality nifti image expected'
        modality_filenames.append(mod_wise_files[0])

    slices = _crop_to_non_zero_content(modality_filenames)
    for mod_fn in modality_filenames:
        data = nib.load(mod_fn)
        data = _crop_img_to(data, slices, copy=True)
        data, _ = _resize_nifti_images(data, interpolation='linear', resized_img_shape=resized_img_shape)
        modality_images.append(data)
    modality_images = np.asarray(modality_images)

    std = np.std(modality_images, axis=(-3, -2, -1))
    mean = np.mean(modality_images, axis=(-3, -2, -1))
    return mean, std


def _generate_samples(path_brats, val_split=0.8, resized_img_shape=(144, 144, 144), bias_correction=False):

    path_brats_data = os.path.join(path_brats, 'data')
    samples = glob(os.path.join(path_brats_data, 'LGG', '*')) + glob(os.path.join(path_brats_data, 'HGG', '*'))

    path_brats_bias_corrected = os.path.join(path_brats, 'bias_corrected')
    path_brats_preprocessed = os.path.join(path_brats, 'preprocessed')

    num_cpu = mp.cpu_count()
    pool = Pool(processes=num_cpu)

    if bias_correction is True:
        if not os.path.exists(path_brats_bias_corrected):
            print('Start: Bias Correction Step')
            pool.map(_generate_bias_corrected, samples)
            print("End: Bias Correction step")

    if bias_correction is True:
        samples = glob(os.path.join(path_brats_bias_corrected, 'LGG', '*')) + \
                    glob(os.path.join(path_brats_bias_corrected, 'HGG', '*'))
    else:
        samples = glob(os.path.join(path_brats_data, 'LGG', '*')) + \
                    glob(os.path.join(path_brats_data, 'HGG', '*'))

    if not os.path.exists(path_brats_preprocessed):
        os.mkdir(path_brats_preprocessed)

        resized_img_shape_list = [resized_img_shape] * len(samples)
        results = pool.starmap(_generate_mean_std_from_sample, zip(samples, resized_img_shape_list))
        mean_sample_wise = []
        std_sample_wise = []
        for mean_item, std_item in results:
            mean_sample_wise.append(mean_item)
            std_sample_wise.append(std_item)
        mean_sample_wise = np.asarray(mean_sample_wise)
        std_sample_wise = np.asarray(std_sample_wise)
        std_samples = np.mean(std_sample_wise, axis=0, dtype=np.float32)
        mean_samples = np.mean(mean_sample_wise, axis=0, dtype=np.float32)

        path_brats_preprocessed = [path_brats_preprocessed] * len(samples)
        mean_samples = [mean_samples] * len(samples)
        std_samples = [std_samples] * len(samples)
        resized_img_shape_list = [resized_img_shape] * len(samples)
        pool.starmap(_generate_preprocessed_samples_core,
                     zip(path_brats_preprocessed, samples, mean_samples, std_samples, resized_img_shape_list))

    pool.close()
    pool.join()

    train_val_indices = np.arange(len(samples))
    shuffle(train_val_indices)
    n_training = int(len(samples) * val_split)
    train_indices = train_val_indices[:n_training]
    val_indices = train_val_indices[n_training:]

    hgg_samples = glob(os.path.join(path_brats, 'data', 'HGG', '*'))
    lgg_samples = glob(os.path.join(path_brats, 'data', 'LGG', '*'))
    samples = lgg_samples + hgg_samples

    samples_train = [samples[idx] for idx in train_indices]
    train_csv = _create_csv(path_brats, 'train', samples_train)

    samples_val = [samples[idx] for idx in val_indices]
    val_csv = _create_csv(path_brats, 'val', samples_val)
    return train_csv, val_csv


def _crop_to_non_zero_content(modality_filenames):
    tolerance = 1e-5
    background = 0
    # determines non zero region across t1,t2,t1ce,flair images, crops the non-zero region
    data_valid = None
    for idx, mod_fn in enumerate(modality_filenames):
        data_nifti = nib.load(mod_fn)
        data = data_nifti.get_data()
        if idx == 0:
            data_valid = np.zeros(data.shape)
        data_valid_idx = np.logical_or(data < (background - tolerance), data > (background + tolerance))
        data_valid[data_valid_idx] = 1
    coords = np.array(np.where(data_valid))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data_valid.shape)
    slices = [slice(s, e) for s, e in zip(start, end)]
    return slices


def load_data(path_brats=None, resized_img_shape=(144, 144, 144), bias_correction=False):
    if path_brats is None:
        path_brats = os.path.join(str(Path.home()), 'fastestimator_data', 'BraTs')
    train_csv, val_csv = _generate_samples(path_brats, resized_img_shape=resized_img_shape, bias_correction=bias_correction)
    return train_csv, val_csv, path_brats
