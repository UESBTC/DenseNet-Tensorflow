# -*- coding: UTF-8 -*-
import os
import sys
import pickle
import numpy as np
import cv2 as cv
import random

dataset_dir = './cifar-100-python/'
img_size = 32
channel_num = 3


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(filename, lbl_cnt):
    data_dict = unpickle(dataset_dir + filename)
    data = data_dict[b'data']
    labels = data_dict[b'fine_labels']
    data = data.reshape(-1, channel_num, img_size, img_size)
    data = data.transpose(0, 2, 3, 1)
    labels = [[(int)(i == label) for i in range(lbl_cnt)] for label in labels]
    # print(data.dtype)
    return data, np.array(labels, dtype=np.int)


def prepare_data():
    if not os.path.exists(dataset_dir):
        print('Fail to load data')
        sys.exit(1)
    print('Loading data...')
    label_dict = unpickle(dataset_dir + 'meta')
    label_count = len(label_dict[b'fine_label_names'])
    train_data, train_labels = load_data('train', label_count)
    test_data, test_labels = load_data('test', label_count)
    ind = np.random.permutation(train_data.shape[0])
    train_data = train_data[ind]
    train_labels = train_labels[ind]
    print('Load data succesfully')
    print('Train data ' + str(train_data.shape))
    print('Train label ' + str(train_labels.shape))
    print('Test data ' + str(test_data.shape))
    print('Test label ' + str(test_labels.shape))

    return train_data, train_labels, test_data, test_labels


## for data augmentation
def data_augmentation(data, padding=None):
    if padding:
        data = random_crop(data)
    data = random_flip_left_right(data)
    return data


def random_crop(data, crop_shape):
    for item in data:
        npad = ((padding, padding), (padding, padding), (0, 0))
        item = np.liblpad(item, pad_width=npad, mode=constant, constant_values=0)
        nh = random.randint(0, padding)
        nw = random.randint(0, padding)
        item = item[nh:nh + crop_shape[0], nw:nw + crop_shape[1], :]
    return data


def random_flip_left_right(data):
    for item in data:
        if bool(random.getrandbits(1)):
            item = np.fliplr(item)
    return data


def color_preprocess(train_data, test_data):
    train_data[..., 0] = ((train_data[..., 0] - np.mean(train_data[..., 0])) / np.std(train_data[..., 0]))
    train_data[..., 1] = ((train_data[..., 1] - np.mean(train_data[..., 1])) / np.std(train_data[..., 1]))
    train_data[..., 2] = ((train_data[..., 2] - np.mean(train_data[..., 2])) / np.std(train_data[..., 2]))
    test_data[..., 0] = ((test_data[..., 0] - np.mean(test_data[..., 0])) / np.std(test_data[..., 0]))
    test_data[..., 1] = ((test_data[..., 1] - np.mean(test_data[..., 1])) / np.std(test_data[..., 1]))
    test_data[..., 2] = ((test_data[..., 2] - np.mean(test_data[..., 2])) / np.std(test_data[..., 2]))
    return train_data,test_data


if __name__ == '__main__':
    prepare_data()
