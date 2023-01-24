from dataset.data import DataSet
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from random import randint, sample

path_to_dir = "/home/nikhil/MyWork/OwnWork/SegNet/CamVid/content/SegNet-Tutorial/CamVid/"
data = DataSet(path_to_dir=path_to_dir)
Dataset = data.create_dataset()
X_train = Dataset['train'][0]
y_train = Dataset['train'][1]

X_test = Dataset['test'][0]
y_test = Dataset['test'][1]

print(X_test.shape, len(y_test))
print(X_train.dtype, y_train.dtype)
