import numpy as np
import argparse
from dataset.data import DataSet
from segnet.model import SegNet

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Path to dataset')
parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
args = vars(parser.parse_args())


def train(path_to_dir=None,
          batch_size=10,
          epochs=10,
          optimizer='sgd',
          n_labels=12):
    data = DataSet(path_to_dir=path_to_dir)
    data_set = data.create_dataset()
    X_train = data_set['train'][0]
    y_train = data_set['train'][1]

    X_val = data_set['val'][0]
    y_val = data_set['val'][1]

    X_test = data_set['test'][0]
    y_test = data_set['test'][1]

    # TODO test test set
    model = SegNet(output_dim=n_labels)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=[X_val, y_val], batch_size=batch_size,
                        shuffle=True)
    return history


if __name__ == '__main__':
    train(
        path_to_dir=args['path'],
        batch_size=args['batch_size'],
        epochs=args['epochs']
    )
