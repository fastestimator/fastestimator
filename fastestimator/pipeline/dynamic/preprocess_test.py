from unittest import TestCase
from fastestimator.pipeline.dynamic.preprocess import Zscore, Minmax, Scale, Onehot, Resize, Reshape, NrrdReader, DicomReader, ImageReader
import numpy as np
import os

eps = 1e-4
shape = (5, 5, 3)
data = 255 * np.eye(5)
data = np.expand_dims(data, 2)
img = np.repeat(data, 3, 2)

labels = np.array([2])
scalar = 0.2
num_dims = 3
resize_shape = np.array([3, 3])
reshape_shape = (shape[1], shape[0] * shape[2])

img_zscore = np.array([[ 2. , -0.5, -0.5, -0.5, -0.5],
                       [-0.5,  2. , -0.5, -0.5, -0.5],
                       [-0.5, -0.5,  2. , -0.5, -0.5],
                       [-0.5, -0.5, -0.5,  2. , -0.5],
                       [-0.5, -0.5, -0.5, -0.5,  2. ]], dtype=np.float32)
img_zscore = np.expand_dims(img_zscore, 2)
img_zscore = np.repeat(img_zscore, 3, 2)

img_minmax = np.eye(5, dtype=np.float32)
img_minmax = np.expand_dims(img_minmax, 2)
img_minmax = np.repeat(img_minmax, 3, 2)

img_scale = 51 * np.eye(5, dtype=np.float32)
img_scale = np.expand_dims(img_scale, 2)
img_scale = np.repeat(img_scale, 3, 2)

labels_onehot = np.array([[0., 0., 1.]])

img_resize = np.array([[141.66666667, 0., 0.],
                       [0., 255., 0.],
                       [0., 0., 141.66666667]], dtype=np.float32)
img_resize = np.expand_dims(img_resize, 2)
img_resize = np.repeat(img_resize, 3, 2)

img_reshape = np.array([[255., 255., 255.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0., 255., 255., 255.,   0.,   0.,   0.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0., 255., 255., 255.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 255., 255.,
                        255.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                          0., 255., 255., 255.]])

class TestNrrdReader(TestCase):
    def test_transform(self):
        import nibabel
        nib = nibabel.Nifti1Image(img, np.eye(4))
        filename = 'nrrd.nii.gz'
        nib.to_filename(filename)
        preprocess = NrrdReader()
        transformed_data = preprocess.transform(filename)
        assert np.allclose(transformed_data, img, rtol=eps)
        os.remove(filename)


class TestDicomReader(TestCase):
    def test_transform(self):
        filename = 'img.dcm'
        data = 255 * np.ones((1389, 472))
        ds = self._add_dicom_data(data)
        ds.save_as(filename)
        preprocess = DicomReader()
        transformed_data = preprocess.transform(filename)
        assert np.allclose(transformed_data, data, rtol=eps)
        os.remove(filename)

    def _add_dicom_data(self, data):
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ImplicitVRLittleEndian
        import datetime, time
        filename = 'img.dcm'
        pixel_array = data

        filename_endian = os.path.join(os.curdir, filename)
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'

        ds = FileDataset(filename_endian, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.ContentTime = str(time.time())  # milliseconds since the epoch
        ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 3.6.5'

        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = '\\x00\\x00'
        ds.LargestImagePixelValue = '\\xff\\xff'
        ds.Columns = pixel_array.shape[0]
        ds.Rows = pixel_array.shape[1]
        if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
        ds.PixelData = pixel_array.tostring()

        ds.SmallestImagePixelValue = pixel_array.min()
        ds[0x00280106].VR = 'US'
        ds.LargestImagePixelValue = pixel_array.max()
        ds[0x00280107].VR = 'US'
        return ds


class TestImageReader(TestCase):
    def test_transform(self):
        from cv2 import imwrite
        filename = 'img.png'
        imwrite(filename, img)
        preprocess = ImageReader()
        transformed_data = preprocess.transform(filename)
        assert np.allclose(transformed_data, img, rtol=eps)
        os.remove(filename)


class TestZscore(TestCase):
    def test_transform(self):
        preprocess = Zscore()
        transformed_data = preprocess.transform(img)
        assert np.allclose(transformed_data, img_zscore, rtol=eps)


class TestMinmax(TestCase):
    def test_transform(self):
        preprocess = Minmax()
        transformed_data = preprocess.transform(img)
        assert np.allclose(transformed_data, img_minmax, rtol=eps)


class TestScale(TestCase):
    def test_transform(self):
        preprocess = Scale(scalar)
        transformed_data = preprocess.transform(img)
        assert np.allclose(transformed_data, img_scale, rtol=eps)


class TestOnehot(TestCase):
    def test_transform(self):
        preprocess = Onehot(num_dims)
        transformed_data = preprocess.transform(labels)
        assert np.allclose(transformed_data, labels_onehot, rtol=eps)


class TestReshape(TestCase):
    def test_transform(self):
        preprocess = Reshape(reshape_shape)
        transformed_data = preprocess.transform(img)
        assert np.allclose(transformed_data, img_reshape, rtol=eps)

class TestResize(TestCase):
    def test_transform(self):
        preprocess = Resize(resize_shape)
        transformed_data = preprocess.transform(img)
        assert np.allclose(transformed_data, img_resize, rtol=eps)
