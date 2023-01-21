import tensorflow as tf
from sklearn.datasets import load_sample_images
from tensorflow import keras
from keras.utils import conv_utils
from keras.layers import Layer
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
