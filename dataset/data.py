from segnet.model import SegNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from matplotlib.gridspec import GridSpec
from random import randint, sample
import pickle
import tensorflow as tf


class DataSet:
    def __init__(self,
                 path_to_dir=''
                 ):
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
        self.create_dataset()
        return

    def create_dataset(self):
        # TODO
        print(f'Processed {len(self.training_images)} training images.')
        print(f'Processed {len(self.valid_images)} validation set images.')
        print(f'Processed {len(self.valid_images)} testing set images.')
