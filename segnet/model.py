from abc import ABC

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Input, Reshape, BatchNormalization, Activation, Dense, Layer
from keras.models import Model


# from ..segnet.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


class ConvolutionBlock(Layer):
    def __init__(self,
                 filters=64,
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 strides=(1, 1),
                 padding='SAME',
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.conv_1 = Conv2D(filters=filters, kernel_size=self.kernel_size, strides=self.strides,
                             padding=self.padding)
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')

    def call(self, inputs):
        x = inputs
        x = self.conv_1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class EncoderDecoder(Layer):
    def __init__(self,
                 conv_blocks=2,
                 n_filters=[64, 64],
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 pool_size=(2, 2),
                 padding='SAME',
                 **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.conv_blocks = conv_blocks
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.padding = padding

        self.blocks = []
        for i in range(conv_blocks):
            self.blocks.append(ConvolutionBlock(n_filters[i], kernel_size=self.kernel_size, pool_size=self.pool_size,
                                                strides=self.strides, padding=self.padding))

    def call(self, inputs):
        x = inputs
        for layer in self.blocks:
            x = layer(x)
        return x


class SegNet(Model, ABC):
    def __init__(self, output_dim=12, **kwargs):
        super(SegNet, self).__init__(**kwargs)

