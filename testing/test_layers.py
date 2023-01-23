import tensorflow as tf
from sklearn.datasets import load_sample_images
from segnet.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from segnet.layers import ConvolutionBlock, EncoderDecoder

images = load_sample_images()["images"]
images = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 255)(images)

max_layer = MaxPoolingWithArgmax2D()
un_pool_layer = MaxUnpooling2D()
x, i = max_layer(images)
print(x.shape)
x = un_pool_layer([x, i])
print(x.shape)

x = ConvolutionBlock(input_shape=(427, 640, 3))
images = load_sample_images()["images"]
images = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 255)(images)
res = x(images)
print(res.shape)  # Output should be [2, 427, 640, 64]

y = EncoderDecoder(n_filters=[64, 128])
res = y(images)
print(res.shape)  # Output should be [2, 427, 640, 128]
