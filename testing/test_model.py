from sklearn.datasets import load_sample_images
import tensorflow as tf

from ..segnet.model import SegNet

# build model on batch size of 10 and image size 224x224x3
model = SegNet(12)
model.build(input_shape=(10, 224, 224, 3))
model.summary()