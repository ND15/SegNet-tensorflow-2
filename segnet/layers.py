import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Layer
from keras import backend as k

"""
    Default parameters are as specified in the paper
    Paper: https://arxiv.org/abs/1511.00561
"""


class MaxPoolingWithArgmax2D(Layer):
    """
    SegNet Implementation for MaxPooling extended from tf.nn.max_pool_with_argmax
    """

    def __init__(self,
                 pool_size=(2, 2),
                 strides=(2, 2),
                 padding="SAME",
                 **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, x, **kwargs):
        output, indices = tf.nn.max_pool_with_argmax(
            input=x,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=self.strides,
            padding=self.padding,
        )
        indices = k.cast(indices, dtype=k.floatx())
        return [output, indices]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def get_config(self):
        config = super(MaxPoolingWithArgmax2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class MaxUnpooling2D(Layer):
    """
    SegNet Implementation for UpSampling layer

    Code references:
        https://github.com/PavlosMelissinos/enet-keras/blob/master/src/models/layers/pooling.py
        https://github.com/ykamikawa/tf-keras-SegNet/blob/master/layers.py
        https://github.com/mvoelk/keras_layers/blob/master/layers.py
    """

    def __init__(self,
                 size=(2, 2),
                 **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, indices = inputs[0], inputs[1]
        indices = tf.cast(indices, dtype='int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (input_shape[0],
                            input_shape[1] * self.size[0],
                            input_shape[2] * self.size[1],
                            input_shape[3])

        one_like_mask = k.ones_like(indices, dtype='int32')
        batch_shape = k.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = k.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range

        y = indices // (output_shape[2] * output_shape[3])
        x = (indices // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = k.transpose(k.reshape(k.stack([b, y, x, f]), [4, updates_size]))
        values = k.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'size': self.size,
        })
        return config


class ConvolutionBlock(Layer):
    def __init__(self,
                 filters=64,
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 strides=(1, 1),
                 padding='SAME',
                 **kwargs):
        """
            Each block consists of Convolution -> BatchNormalization -> Activation
        """
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
        """
            Multiple convolution blocks are treated as encoders or decoders
        """
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

    def get_config(self):
        config = super(EncoderDecoder, self).get_config()
        config.update({
            'conv_blocks': self.conv_blocks,
            'n_filters': self.n_filters,
            'strides': self.strides,
            'pool_size': self.pool_size,
            'padding': self.padding
        })
