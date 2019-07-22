import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal
from cyclegan_components import InstanceNormalization, ReflectionPadding2D

class Network(object):
    def __init__(self, LAMBDA=10.0):
        self.LAMBDA = LAMBDA
        self.generator = {
            "XtoY": self._build_generator(),
            "YtoX": self._build_generator()
        }

        self.discriminator = {
            "X": self._build_discriminator(),
            "Y": self._build_discriminator()
        }
        
        self.optimizers = {
            "XtoY": tf.keras.optimizers.Adam(2e-4, 0.5),
            "YtoX": tf.keras.optimizers.Adam(2e-4, 0.5),
            "X": tf.keras.optimizers.Adam(2e-4, 0.5),
            "Y": tf.keras.optimizers.Adam(2e-4, 0.5)
        }
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.loss_obj = tf.keras.losses.MeanSquaredError()

    def _resblock(self, x0, num_filter=256, kernel_size=3):
        x = ReflectionPadding2D()(x0)
        x = layers.Conv2D(filters=num_filter,
                        kernel_size=kernel_size,
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(filters=num_filter,
                        kernel_size=kernel_size, 
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.Add()([x, x0])
        return x

    def _build_discriminator(self, input_shape=(256, 256, 3)):        
        x0 = layers.Input(input_shape)
        x = layers.Conv2D(filters=64,
                        kernel_size=4,
                        strides=2,
                        padding='same',
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x0)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(filters=128,
                        kernel_size=4,
                        strides=2,
                        padding='same',
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(filters=256,
                        kernel_size=4,
                        strides=2,
                        padding='same',    
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)    
        
        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(filters=512,
                        kernel_size=4,
                        strides=1,
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(filters=1,
                        kernel_size=4,
                        strides=1,
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

        return Model(inputs=x0, outputs=x)

    def _build_generator(self, input_shape=(256, 256, 3), num_blocks=9):
        x0 = layers.Input(input_shape)

        x = ReflectionPadding2D(padding=(3, 3))(x0)
        x = layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=1,
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

        # downsample
        x = layers.Conv2D(filters=128,
                        kernel_size=3, 
                        strides=2,
                        padding='same',                    
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters=256,
                        kernel_size=3, 
                        strides=2,
                        padding='same',                    
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)    
        
        # residual     
        for _ in range(num_blocks):
            x = self._resblock(x)
        
        # upsample
        x = layers.Conv2DTranspose(filters=128,
                                kernel_size=3,
                                strides=2,
                                padding='same',                            
                                kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(filters=64,
                                kernel_size=3,
                                strides=2,
                                padding='same', 
                                kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

        # final
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(filters=3,
                        kernel_size=7,
                        activation='tanh',
                        kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
        
        return Model(inputs=x0, outputs=x)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss
    
    def _train_step(self, real_x, real_y, buffer_size=50):
        loss = {}
        pred = {}
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            fake_y = self.generator["XtoY"](real_x, training=True)
            cycled_x = self.generator["YtoX"](fake_y, training=True)

            fake_x = self.generator["YtoX"](real_y, training=True)
            cycled_y = self.generator["XtoY"](fake_x, training=True)                
            
            # same_x and same_y are used for identity loss.
            same_x = self.generator["YtoX"](real_x, training=True)
            same_y = self.generator["XtoY"](real_y, training=True)

            disc_real_x = self.discriminator["X"](real_x, training=True)
            disc_real_y = self.discriminator["Y"](real_y, training=True)
            
            disc_fake_x = self.discriminator["X"](fake_x, training=True)
            disc_fake_y = self.discriminator["Y"](fake_y, training=True)        
            # calculate the loss
            gen_XtoY_loss = self.generator_loss(disc_fake_y)
            gen_YtoX_loss = self.generator_loss(disc_fake_x)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_XtoY_loss = gen_XtoY_loss + self.cycle_loss(real_x, cycled_x) + self.identity_loss(real_x, same_x)
            total_gen_YtoX_loss = gen_YtoX_loss + self.cycle_loss(real_y, cycled_y) + self.identity_loss(real_y, same_y) 
            if len(self.fake_x_list) < buffer_size:
                self.fake_x_list.append(fake_x)
                self.fake_y_list.append(fake_y)
            else:
                if random.random() > 0.5:
                    idx_ = random.randint(0, buffer_size-1)                
                    (self.fake_x_list[idx_], fake_x) = (fake_x, self.fake_x_list[idx_])
                    (self.fake_y_list[idx_], fake_y) = (fake_y, self.fake_y_list[idx_])
                    disc_fake_x = self.discriminator["X"](fake_x, training=True)
                    disc_fake_y = self.discriminator["Y"](fake_y, training=True)                               
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        grad_gen_x2y = gen_tape.gradient(total_gen_XtoY_loss, self.generator["XtoY"].trainable_variables)
        grad_gen_y2x = gen_tape.gradient(total_gen_YtoX_loss, self.generator["YtoX"].trainable_variables)
        grad_disc_x = disc_tape.gradient(disc_x_loss, self.discriminator["X"].trainable_variables)
        grad_disc_y = disc_tape.gradient(disc_y_loss, self.discriminator["Y"].trainable_variables)

        self.optimizers["XtoY"].apply_gradients(zip(grad_gen_x2y, self.generator["XtoY"].trainable_variables))
        self.optimizers["YtoX"].apply_gradients(zip(grad_gen_y2x, self.generator["YtoX"].trainable_variables))            
        self.optimizers["X"].apply_gradients(zip(grad_disc_x, self.discriminator["X"].trainable_variables))
        self.optimizers["Y"].apply_gradients(zip(grad_disc_y, self.discriminator["Y"].trainable_variables))

        pred["X_fake"] = fake_x
        pred["Y_fake"] = fake_y
        pred["X_fake_dis"] = disc_fake_x
        pred["X_real_dis"] = disc_real_x
        pred["Y_fake_dis"] = disc_fake_y
        pred["Y_real_dis"] = disc_real_y

        loss["XtoY"] = total_gen_XtoY_loss
        loss["YtoX"] = total_gen_YtoX_loss
        loss["X"] = disc_x_loss
        loss["Y"] = disc_y_loss
        return pred, loss

    def train_op(self, features):
        real_x = features["img_X"]
        real_y = features["img_Y"]
        self.fake_x_list = []
        self.fake_y_list = []
        pred, loss = self._train_step(real_x, real_y)
        return pred, loss

    def eval_op(self, features):
        real_x = features["img_X"]
        real_y = features["img_Y"]
        
        fake_y = self.generator["XtoY"](real_x, training=True)
        cycled_x = self.generator["YtoX"](fake_y, training=True)
        # same_x and same_y are used for identity loss.
        same_x = self.generator["YtoX"](real_x, training=True)

        disc_real_y = self.discriminator["Y"](real_y, training=True)
        disc_fake_y = self.discriminator["Y"](fake_y, training=True)        

        # calculate the loss
        gen_XtoY_loss = self.generator_loss(disc_fake_y)
        # Total generator loss = adversarial loss + cycle loss
        total_gen_XtoY_loss = gen_XtoY_loss + self.cycle_loss(real_x, cycled_x) + self.identity_loss(real_x, same_x)
            
        disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        loss = dict()
        loss["XtoY"] = total_gen_XtoY_loss
        loss["Y"] = disc_y_loss

        pred = dict()
        pred["Y_fake"] = fake_y
        pred["Y_fake_dis"] = disc_fake_y
        pred["Y_real_dis"] = disc_real_y
        
        return pred, total_gen_XtoY_loss
