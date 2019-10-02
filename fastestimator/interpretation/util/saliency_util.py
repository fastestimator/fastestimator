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
import tensorflow as tf


class SaliencyMask(object):
    """Base class for saliency masks. Alone, this class doesn't do anything.
    This code was adapted from https://github.com/PAIR-code/saliency to be compatible with TensorFlow 2.0
    """
    def __init__(self, model):
        """
        Args:
            model: The ML model to evaluate masks on
        """
        self.model = model

    @tf.function
    def get_mask(self, model_input):
        """
        Args:
            model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
        Returns:
            A saliency mask
        """
        raise NotImplementedError('A derived class should implemented GetMask()')

    def get_smoothed_mask(self, model_input, stdev_spread=.15, nsamples=25, magnitude=True, **kwargs):
        """
        Args:
            model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
            stdev_spread: Amount of noise to add to the input, as fraction of the
                        total spread (x_max - x_min). Defaults to 15%.
            nsamples: Number of samples to average across to get the smooth gradient.
            magnitude: If true, computes the sum of squares of gradients instead of
                     just the sum. Defaults to true.
        Returns:
            A saliency mask that is smoothed with the SmoothGrad method
        """
        stdev = stdev_spread * (tf.reduce_max(model_input) - tf.reduce_min(model_input))

        # Adding noise to the image might cause the max likelihood class value to change, so need to keep track of
        # which class we're comparing to
        initial_predictions = self.model(model_input)
        class_indices = tf.reshape(tf.argmax(initial_predictions, 1, output_type='int64'), (model_input.shape[0], 1))
        row_indices = tf.reshape(tf.range(class_indices.shape[0], dtype='int64'), (class_indices.shape[0], 1))
        classes = tf.concat([row_indices, class_indices], 1)

        total_gradients = tf.zeros_like(model_input, dtype='float32')
        for _ in tf.range(nsamples):
            noise = tf.random.normal(model_input.shape, mean=0, stddev=stdev)
            x_plus_noise = model_input + noise
            grad = self.get_mask(x_plus_noise, classes=classes, **kwargs)
            if magnitude:
                total_gradients = tf.add(tf.multiply(grad, grad), total_gradients)
            else:
                total_gradients = tf.add(grad, total_gradients)

        return total_gradients / nsamples


class GradientSaliency(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with a gradient."""
    def __init__(self, model):
        super().__init__(model)

    @tf.function
    def get_mask(self, model_input, classes=None):
        """
        Args:
            model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
            classes: A tensor describing which output index to consider for each element in the input batch.
                    ex. [[0, 5], [1, 7]] for an input with batch size 2 where indexes 5 and 7 are the targets
        Returns:
            A vanilla gradient mask
        """
        with tf.GradientTape() as tape:
            tape.watch(model_input)
            logits = self.model(model_input, training=False)
            if classes is None:
                y = tf.reduce_max(logits, 1)  # The maximum likelihood class scores
            else:
                y = tf.gather_nd(logits, classes)
        return tape.gradient(y, model_input)


class IntegratedGradients(GradientSaliency):
    """A SaliencyMask class that implements the integrated gradients method.

    https://arxiv.org/abs/1703.01365
    """
    @tf.function
    def get_mask(self, model_input, input_baseline=None, steps=25, classes=None):
        """
        Args:
            model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
            input_baseline: Baseline value used in integration. Defaults to 0.
            steps: Number of integrated steps between baseline and image.
            classes: A tensor describing which output index to consider for each element in the input batch.
                    ex. [[0, 5], [1, 7]] for an input with batch size 2 where indexes 5 and 7 are the targets
        Returns:
            An integrated gradients mask
        """
        if input_baseline is None:
            input_baseline = tf.zeros_like(model_input, dtype=model_input.dtype)

        assert input_baseline.shape == model_input.shape

        x_diff = model_input - input_baseline

        total_gradients = tf.zeros_like(model_input, dtype='float32')

        # Performing this integration might cause the max likelihood class value to change, so need to keep track of
        # which class we're comparing to
        if classes is None:
            initial_predictions = self.model(model_input)
            class_indices = tf.reshape(tf.argmax(initial_predictions, 1, output_type='int64'),
                                       (model_input.shape[0], 1))
            row_indices = tf.reshape(tf.range(class_indices.shape[0], dtype='int64'), (class_indices.shape[0], 1))
            classes = tf.concat([row_indices, class_indices], 1)

        for alpha in tf.linspace(0.0, 1.0, steps):
            x_step = input_baseline + alpha * x_diff
            total_gradients += super().get_mask(x_step, classes=classes)

        return tf.multiply(total_gradients, x_diff)
