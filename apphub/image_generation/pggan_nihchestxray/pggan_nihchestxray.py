import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend

import fastestimator as fe
from fastestimator.dataset.nih_chestxray import load_data
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.numpyop import Resize as ResizeRecord
from fastestimator.op.tensorop import Loss, ModelOp, Reshape, Resize
from fastestimator.schedule import Scheduler
from fastestimator.trace import Trace
from fastestimator.util.record_writer import RecordWriter
from pggan_architecture import build_D, build_G


class Rescale(TensorOp):
    """Scale image values from uint8 to float32 between -1 and 1."""
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


class CreateLowRes(TensorOp):
    def forward(self, data, state):
        data_shape = tf.shape(data)
        height = data_shape[0]
        width = data_shape[1]
        data = tf.image.resize(data, (height / 2, width / 2))
        data = tf.image.resize(data, (height, width))
        return data


class RandomInput(TensorOp):
    def forward(self, data, state):
        latent_dim = data
        batch_size = state["local_batch_size"]
        random_vector = tf.random.normal([batch_size, latent_dim])
        return random_vector


class ImageBlender(TensorOp):
    def __init__(self, alpha, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha

    def forward(self, data, state):
        image, image_lowres = data
        new_img = self.alpha * image + (1 - self.alpha) * image_lowres
        return new_img


class Interpolate(TensorOp):
    def forward(self, data, state):
        fake, real = data
        batch_size = state["local_batch_size"]
        coeff = tf.random.uniform(shape=[batch_size, 1, 1, 1],
                                  minval=0.0,
                                  maxval=1.0,
                                  dtype=tf.float32)
        return real + (fake - real) * coeff


class GradientPenalty(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        x_interp, interp_score = data
        interp_score = tf.reshape(interp_score, [-1])
        tape = state['tape']
        gradient_x_interp = tape.gradient(tf.reduce_sum(interp_score), x_interp)
        grad_l2 = tf.math.sqrt(tf.reduce_sum(tf.math.square(gradient_x_interp), axis=[1, 2, 3]))
        gp = tf.math.square(grad_l2 - 1.0)
        return gp


class GLoss(Loss):
    def forward(self, data, state):
        return -data


class DLoss(Loss):
    """Compute discriminator loss."""
    def __init__(self, inputs, outputs=None, mode=None, wgan_lambda=10, wgan_epsilon=0.001):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.wgan_lambda = wgan_lambda
        self.wgan_epsilon = wgan_epsilon

    def forward(self, data, state):
        real_score, fake_score, gp = data
        loss = fake_score - real_score + self.wgan_lambda * gp + tf.math.square(real_score) * self.wgan_epsilon
        return loss


class AlphaController(Trace):
    def __init__(self, alpha, fade_start, duration):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.alpha = alpha
        self.fade_start = fade_start
        self.duration = duration
        self.change_alpha = False
        self._idx = 0

    def on_epoch_begin(self, state):
        # check whetehr the current epoch is in smooth transition of resolutions
        fade_epoch = self.fade_start[self._idx]
        if state["epoch"] == fade_epoch:
            self.nimg_total = self.duration[self._idx] * state["num_examples"]
            self.change_alpha = True
            self.nimg_so_far = 0
            print("FastEstimator-Alpha: Started fading in for size {}".format(2**(self._idx+3)))
        elif state["epoch"] == fade_epoch + self.duration[self._idx]:
            print("FastEstimator-Alpha: Finished fading in for size {}".format(2**(self._idx+3)))
            self.change_alpha = False
            self._idx += 1
            backend.set_value(self.alpha, 1.0)

    def on_batch_begin(self, state):
        # if in resolution transition, smoothly change the alpha from 0 to 1
        if self.change_alpha:
            self.nimg_so_far += state["batch_size"]
            current_alpha = np.float32(self.nimg_so_far / self.nimg_total)
            backend.set_value(self.alpha, current_alpha)

    # def on_batch_end(self, state):
    #     if state["train_step"] % 10 == 0:
    #         tf.print(self.alpha)


class ImageSaving(Trace):
    def __init__(self, epoch_model, save_dir, num_sample=16, latent_dim=512):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.epoch_model = epoch_model
        self.save_dir = save_dir
        self.latent_dim = latent_dim
        self.num_sample = num_sample
        self.random_vectors = tf.random.normal(
            [self.num_sample, self.latent_dim])

    def on_epoch_end(self, state):
        if state["epoch"] in self.epoch_model:
            model_name = self.epoch_model[state["epoch"]]
            model = self.network.model[model_name]
            pred = model(self.random_vectors)
            eps = 1e-8
            fig = plt.figure(figsize=(4, 4))
            for i in range(pred.shape[0]):
                plt.subplot(4, 4, i + 1)
                disp_img = pred[i].numpy()
                disp_img -= disp_img.min()
                disp_img /= (disp_img.max() + eps)
                plt.imshow(disp_img)
            plt.savefig(
                os.path.join(self.save_dir,
                             'image_at_{:08d}.png').format(state["epoch"]))
            print("on epoch {}, saving image to {}".format(
                state["epoch"], self.save_dir))


def get_estimator():
    train_csv, data_path = load_data()
    writer = RecordWriter(save_dir=os.path.join(data_path, "tfrecord"),
                          train_data=train_csv,
                          ops=[
                              ImageReader(inputs="x", parent_path=data_path),
                              ResizeRecord(target_size=(128, 128), outputs="x")
                          ])

    # We create a scheduler for batch_size with the epochs at which it will change and corresponding values.
    batchsize_scheduler = Scheduler({0: 128, 5: 64, 15: 32, 25: 16, 35: 8, 45:4})

    # We create a scheduler for the Resize ops.
    resize_scheduler = Scheduler({
        0: Resize(inputs="x", size=(4, 4), outputs="x"),
        5: Resize(inputs="x", size=(8, 8), outputs="x"),
        15:Resize(inputs="x", size=(16, 16), outputs="x"),
        25:Resize(inputs="x", size=(32, 32), outputs="x"),
        35:Resize(inputs="x", size=(64, 64), outputs="x"),
        45:None
    })

    # In Pipeline, we use the schedulers for batch_size and ops.
    pipeline = fe.Pipeline(
        batch_size=batchsize_scheduler,
        data=writer,
        ops=[
            resize_scheduler,
            CreateLowRes(inputs="x", outputs="x_lowres"),
            Rescale(inputs="x", outputs="x"),
            Rescale(inputs="x_lowres", outputs="x_lowres")
        ])

    opt2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    opt3 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    opt4 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    opt5 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    opt6 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    opt7 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    fade_in_alpha = tf.Variable(initial_value=1.0, dtype='float32', trainable=False)
    d2, d3, d4, d5, d6, d7 = fe.build(model_def=lambda: build_D(fade_in_alpha=fade_in_alpha, target_resolution=7),
                                        model_name=["d2", "d3", "d4", "d5", "d6", "d7"],
                                        optimizer=[opt2, opt3, opt4, opt5, opt6, opt7],
                                        loss_name=["dloss", "dloss", "dloss", "dloss", "dloss", "dloss"])

    g2, g3, g4, g5, g6, g7, G = fe.build(model_def=lambda: build_G(fade_in_alpha=fade_in_alpha, target_resolution=7),
                                        model_name=["g2", "g3", "g4", "g5", "g6", "g7", "G"],
                                        optimizer=[opt2, opt3, opt4, opt5, opt6, opt7, opt7],
                                        loss_name=["gloss", "gloss", "gloss", "gloss", "gloss", "gloss", "gloss"])

    g_scheduler = Scheduler({
        0: ModelOp(model=g2, outputs="x_fake"),
        5: ModelOp(model=g3, outputs="x_fake"),
        15: ModelOp(model=g4, outputs="x_fake"),
        25: ModelOp(model=g5, outputs="x_fake"),
        35: ModelOp(model=g6, outputs="x_fake"),
        45: ModelOp(model=g7, outputs="x_fake"),
    })

    fake_score_scheduler = Scheduler({
        0: ModelOp(inputs="x_fake", model=d2, outputs="fake_score"),
        5: ModelOp(inputs="x_fake", model=d3, outputs="fake_score"),
        15:ModelOp(inputs="x_fake", model=d4, outputs="fake_score"),
        25:ModelOp(inputs="x_fake", model=d5, outputs="fake_score"),
        35:ModelOp(inputs="x_fake", model=d6, outputs="fake_score"),
        45:ModelOp(inputs="x_fake", model=d7, outputs="fake_score")
    })

    real_score_scheduler = Scheduler({
        0: ModelOp(model=d2, outputs="real_score"),
        5: ModelOp(model=d3, outputs="real_score"),
        15:ModelOp(model=d4, outputs="real_score"),
        25:ModelOp(model=d5, outputs="real_score"),
        35:ModelOp(model=d6, outputs="real_score"),
        45:ModelOp(model=d7, outputs="real_score")
    })

    interp_score_scheduler = Scheduler({
        0: ModelOp(inputs="x_interp", model=d2, outputs="interp_score", track_input=True),
        5: ModelOp(inputs="x_interp", model=d3, outputs="interp_score", track_input=True),
        15: ModelOp(inputs="x_interp", model=d4, outputs="interp_score", track_input=True),
        25: ModelOp(inputs="x_interp", model=d5, outputs="interp_score", track_input=True),
        35: ModelOp(inputs="x_interp", model=d6, outputs="interp_score", track_input=True),
        45: ModelOp(inputs="x_interp", model=d7, outputs="interp_score", track_input=True)
    })

    network = fe.Network(ops=[
        RandomInput(inputs=lambda: 512),
        g_scheduler,
        fake_score_scheduler,
        ImageBlender(inputs=("x", "x_lowres"), alpha=fade_in_alpha),
        real_score_scheduler,
        Interpolate(inputs=("x_fake", "x"), outputs="x_interp"),
        interp_score_scheduler,
        GradientPenalty(inputs=("x_interp", "interp_score"), outputs="gp"),
        GLoss(inputs="fake_score", outputs="gloss"),
        DLoss(inputs=("real_score", "fake_score", "gp"), outputs="dloss")
    ])

    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline,
        epochs=55,
        traces=[AlphaController(alpha=fade_in_alpha, fade_start=[5, 15, 25, 35, 45, 55], duration=[5, 5, 5, 5, 5, 5]),
                ImageSaving(epoch_model={4: "g2", 14: "g3", 24: "g4", 34: "g5", 44: "g6", 54: "g7"},
                            save_dir="/data/Xiaomeng/images")])
    return estimator
