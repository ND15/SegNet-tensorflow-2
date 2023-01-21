import tensorflow as tf
from sklearn.datasets import load_sample_images
from ..segnet.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

images = load_sample_images()["images"]
images = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 255)(images)

max_layer = MaxPoolingWithArgmax2D()
un_pool_layer = MaxUnpooling2D()
x, i = max_layer(images)
print(x.shape)
x = un_pool_layer([x, i])
print(x.shape)
