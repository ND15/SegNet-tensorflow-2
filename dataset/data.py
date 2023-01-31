from segnet.model import SegNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from matplotlib.gridspec import GridSpec
from random import randint, sample
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def binary_lab(labels):
    x = np.zeros([labels.shape[0], labels.shape[1], 12])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            x[i, j, labels[i][j]] = 1
    return x


def create_sets(image_list, mask_list):
    images = []
    masks = []

    for img, mask in zip(image_list, mask_list):
        images.append(cv2.resize(cv2.imread(img), (224, 224)))  # 224X224
        masks.append(binary_lab(cv2.resize(cv2.imread(mask), (224, 224))))

    images = np.array(images, dtype='float')
    masks = np.array(masks, dtype='float')
    return images, masks


class DataSet:
    def __init__(self,
                 path_to_dir=''
                 ):
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.path_to_dir = path_to_dir

        # train, val and test paths
        self.training_set = path_to_dir + 'train/'
        print(self.training_set)
        self.test_set = path_to_dir + 'test/'
        self.valid_set = path_to_dir + 'val/'

        # annotated files path
        self.train_label_path = self.path_to_dir + 'trainannot/'
        self.val_label_path = self.path_to_dir + 'valannot/'
        self.test_label_path = self.path_to_dir + 'testannot/'

        # load filenames
        self.training_images = sorted(glob(self.training_set + '*.png'))
        self.training_images_labels = sorted(glob(self.train_label_path + '*.png'))

        self.valid_images = sorted(glob(self.valid_set + '*.png'))
        self.valid_images_labels = sorted(glob(self.val_label_path + '*.png'))

        self.test_images = sorted(glob(self.test_set + '*.png'))
        self.test_images_labels = sorted(glob(self.test_label_path + '*.png'))

        assert len(self.training_images) == len(self.training_images_labels), "Incomplete Labels or training images"
        assert len(self.valid_images) == len(self.valid_images_labels), "Incomplete Labels or valid images"

    def create_dataset(self):
        print("Started")
        self.X_train, self.y_train = create_sets(self.training_images, self.training_images_labels)
        self.X_val, self.y_val = create_sets(self.valid_images, self.valid_images_labels)
        self.X_test, self.y_test = create_sets(self.test_images, self.test_images_labels)

        print(f'Processed {len(self.training_images)} training images.')
        print(f'Processed {len(self.valid_images)} validation set images.')
        print(f'Processed {len(self.test_images)} testing set images.')

        return {
            'train': [self.X_train, self.y_train],
            'val': [self.X_val, self.y_val],
            'test': [self.X_test, self.y_test]
        }
