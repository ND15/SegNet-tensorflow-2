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


def to_numpy(image, dtype=tf.int32):
    with tf.compat.v1.Session() as sess:
        image = tf.cast(image, dtype)
        result = sess.run(image)
    return result


def parse_image(img, resize=(224, 224)):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img)
    if resize:
        img = tf.image.resize(img, resize)
    return img


def create_labels(labels):
    x = np.zeros([labels.shape[0], labels.shape[1], 12])
    labels = to_numpy(labels)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            x[i, j, labels[i][j]] = 1
    return x


def create_sets(image_list, mask_list):
    images = []
    masks = []

    if not mask_list:
        for img in image_list:
            img = parse_image(img, resize=(224, 224))
            img = img / 255.0
            img = to_numpy(img, tf.float32)
            images.append(img)
        return np.array(images), []

    for img, mask in zip(image_list, mask_list):
        img = parse_image(img, resize=(224, 224))
        img = img / 255.0
        img = to_numpy(img, tf.float32)

        mask = parse_image(mask, resize=(224, 224))
        images.append(img)  # 224X224
        masks.append(create_labels(mask))

    images = np.array(images)
    masks = np.array(masks)
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
        self.test_label_path = self.path_to_dir + 'valannot'

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
