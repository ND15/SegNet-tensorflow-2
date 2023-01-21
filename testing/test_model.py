from sklearn.datasets import load_sample_images
import tensorflow as tf

from ..segnet.model import ConvolutionBlock, EncoderDecoder

x = ConvolutionBlock(input_shape=(427, 640, 3))
images = load_sample_images()["images"]
images = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 255)(images)
res = x(images)
print(res.shape)  # Output should be [2, 427, 640, 64]

y = EncoderDecoder(n_filters=[64, 128])
res = y(images)
print(res.shape)  # Output should be [2, 427, 640, 128]

