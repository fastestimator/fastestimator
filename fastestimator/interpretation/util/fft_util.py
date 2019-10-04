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
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def gaussian_kernel(mean, std):
    """
    Args:
        mean: The center of the normal distribution
        std: The standard deviation of the normal distribution
    Returns:
        A gaussian kernel defined by the specified normal distribution, expanded for multiplication against a
        3-channel image
    """
    # kernel size should be 6x the std in order to be similar to global op
    size = tf.maximum(2.0, tf.math.ceil(3 * std))
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    kernel = tf.einsum('i,j->ij', vals, vals)
    kernel = kernel / tf.reduce_sum(kernel)
    # Convert the kernel to 3 channel TODO make dynamic
    kernel = tf.stack([kernel, kernel, kernel], axis=2)
    # Expand kernel shape to match conv2d spec
    kernel = kernel[:, :, :, tf.newaxis]
    return kernel


@tf.function
def blur_image(image, std):
    """
    Args:
        image: A 3 channel input image
        std: The standard deviation of the gaussian filter to be applied
    Returns:
        'image' blurred by a gaussian filter
    """
    kernel = gaussian_kernel(0, std)
    return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")


@tf.function
def blur_image_fft(image, std):
    """This method is equivalent to blur_image(), but is slower right now due to a bug in tensorflow related to graph
    construction for the irfft2d function. TODO re-evaluate performance trade-off as new version of TF2 are released

    Args:
        image: An n-channel input image
        std: The standard deviation of the gaussian filter to be applied
    Returns:
        'image' blurred by a gaussian filter
    """
    mean = 0
    # Wiki article on gaussian blurring indicates that a 6s * 6s filter is basically equivalent to global kernel
    half_kernel_size = tf.maximum(2.0, tf.math.ceil(3.0 * std))

    kernel_size = 2 * half_kernel_size + 1  # Kernel size will always be odd
    b, h, w, c = image.shape
    # Convert to batch-channel-h-w
    tf_image = tf.transpose(image, [0, 3, 1, 2])
    fft_length = [h, w]
    # Need to pad the fft size based on the kernel size. Also, the last axis needs to be padded by one if it ends up
    # being odd. The extra pixel will get cropped off at the end
    fft_length_padded = [fft_length[0] + kernel_size, fft_length[1] + kernel_size + (1 + w) % 2]
    ft_image = tf.signal.rfft2d(tf_image, fft_length=fft_length_padded)

    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-half_kernel_size, limit=half_kernel_size + 1, dtype=tf.float32))
    kernel = tf.einsum('i,j->ij', vals, vals)
    kernel = kernel / tf.reduce_sum(kernel)

    big_kernel = tf.eye(num_rows=kernel_size, num_columns=kernel_size, batch_shape=[b, c], dtype=kernel.dtype)
    kernel = tf.matmul(big_kernel, kernel)
    ft_kernel = tf.signal.rfft2d(kernel, fft_length=fft_length_padded)

    ft_image = ft_image * ft_kernel

    tf_image = tf.signal.irfft2d(ft_image)
    # Convert to batch-dim1-dim2-channel
    tf_image = tf.transpose(tf_image, [0, 2, 3, 1])
    # Cast to int for indexing
    half_kernel_size = tf.cast(half_kernel_size, dtype=tf.int32)
    # Crop the extra pixels
    tf_image = tf_image[:, half_kernel_size:h + half_kernel_size, half_kernel_size:w + half_kernel_size, :]
    return tf_image


def rfft2d_freqs(h, w):
    """
    Args:
        h: height of an image
        w: width of an image
    Returns:
        A 2d fourier spectrum corresponding to the given image dimensions
    """
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    fx = np.fft.fftfreq(w)[:w // 2 + 1 + w % 2]
    return np.sqrt(fx * fx + fy * fy)


@tf.function
def fft_vars_to_whitened_im(fft_vars, scale, width):
    """
    Args:
        fft_vars: A tensor of dimension: real/imaginary X batch X channel X height X width
        scale: A matrix which normalizes the energy of an image
        width: The true width of the pixel image (needed to resolve ambiguity introduced by the fourier transform)
    Returns:
        A pixel-space image corresponding to the given input variables, after applying a normalization by 'scale'
    """
    spectrum = tf.complex(fft_vars[0], fft_vars[1])
    spectrum = scale * spectrum  # whiten the fft domain image (de-correlates adjacent pixels)
    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))  # fft domain required width and height to be the last dimensions
    image = image[:, :, :width, :]  # Crop the extra pixel that results if width is odd
    image = image / 4.0  # Make the image develop more slowly / controlled. Not sure why 4 vs any other factor
    return image


@tf.function
def fft_vars_to_im(fft_vars, width):
    """
    Args:
        fft_vars: A tensor of dimension: real/imaginary X batch X channel X height X width
        width: The true width of the pixel image (needed to resolve ambiguity introduced by the fourier transform)
    Returns:
        A pixel-space image corresponding to the given input variables
    """
    spectrum = tf.complex(fft_vars[0], fft_vars[1])
    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))
    image = image[:, :, :width, :]
    image = image / 4.0  # Make the image develop more slowly / controlled. Not sure why 4 vs any other factor
    return image
