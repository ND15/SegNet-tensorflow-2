from abc import ABC

import keras.layers
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Dense

from segnet.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D, EncoderDecoder


class SegNet(Model, ABC):
    def __init__(self,
                 output_dim=12,
                 activation='relu',
                 encoder_blocks=[2, 2, 3, 3, 3],
                 decoder_blocks=[3, 3, 3, 2, 1],
                 encoder_filters=[64, 64, 128, 128, 256, 256, 256,
                                  512, 512, 512, 512, 512, 512],
                 decoder_filters=[512, 512, 512, 512, 512, 256,
                                  256, 256, 128, 128, 64, 64],
                 **kwargs):
        super(SegNet, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.mask_indices = []

        self.encoder = []
        counter = 0
        for i in range(len(encoder_blocks)):
            self.encoder.append(EncoderDecoder(encoder_blocks[i],
                                               encoder_filters[counter:counter + encoder_blocks[i]]))
            counter += encoder_blocks[i]

        self.decoder = []
        counter = 0
        for i in range(len(decoder_blocks)):
            self.decoder.append(EncoderDecoder(decoder_blocks[i],
                                               decoder_filters[counter:counter + decoder_blocks[i]]))
            counter += decoder_blocks[i]

            # max pooling and unpooling depending on the len of the encoder blocks
        self.max_pooling = [
                               MaxPoolingWithArgmax2D()
                           ] * len(encoder_blocks)
        self.max_un_pooling = [
                                  MaxUnpooling2D()
                              ] * len(decoder_blocks)
        self.conv_layer = Conv2D(self.output_dim, (1, 1), padding='VALID')
        self.batch_norm = BatchNormalization()
        self.output_dense = Dense(units=output_dim)
        self.output_activation = Activation('softmax')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.encoder_blocks)):
            x = self.encoder[i](x)
            x, mask = self.max_pooling[i](x)
            self.mask_indices.append(mask)

        for i in range(len(self.decoder_blocks)):
            x = self.max_un_pooling[i]([x, self.mask_indices[len(self.mask_indices) - 1 - i]])
            x = self.decoder[i](x)

        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.output_dense(x)
        x = self.output_activation(x)

        return x
