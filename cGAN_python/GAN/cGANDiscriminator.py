import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from GAN.cGANGenerator import EncoderLayer
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

class EncoderLayer(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides_s = 2,
                 apply_batchnorm=True,
                 add = False,
                 padding_s = 'same'):

        super(EncoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        conv = layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides_s,
                             padding=padding_s,
                             kernel_initializer=initializer,
                             use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None

        if add:
            self.encoder_layer = tf.keras.Sequential([conv])
        elif apply_batchnorm:
            bn = layers.BatchNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])
        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)

"""
The Discriminator is a PatchGAN.
"""

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.encoder_layer_1 = EncoderLayer(filters=64,
                                            kernel_size=4,
                                            apply_batchnorm=False)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()                                
        self.conv = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=4,
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.bn1 = layers.BatchNormalization()                                 
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()                       
        self.last = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=4,
                                           strides=1,
                                           kernel_initializer=initializer)

    def call(self, y):
        """inputs can be generated image. """
        # target = y
        # x = target
        x = self.encoder_layer_1(y)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)

        return x