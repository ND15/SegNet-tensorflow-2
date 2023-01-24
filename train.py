import numpy as np

from dataset.data import DataSet
from segnet.model import SegNet


def train(path_to_dir=None,
          optimizer='sgd',
          n_labels=12):
    data = DataSet(path_to_dir=path_to_dir)
    data_set = data.create_dataset()
    X_train = data_set['train'][0].astype('float')
    y_train = data_set['train'][1].astype('float')

    X_val = data_set['val'][0].astype('float')
    y_val = data_set['val'][1].astype('float')

    X_test = data_set['test'][0].astype('float')
    y_test = data_set['test'][1].astype('float')

    X_train = np.concatenate([X_train, X_val])
    y_train = np.concatenate([y_train, y_val])

    model = SegNet(12)
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True)
    return history


if __name__ == '__main__':
    train(path_to_dir='/home/nikhil/MyWork/OwnWork/SegNet/CamVid/content/SegNet-Tutorial/CamVid/')
