"""CIFAR100 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from six.moves import cPickle
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os

def my_load_batch(fpath, classes, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    # [print(l) for (l,d)in zip(labels,data)]
    # print(classes)
    # filtered = [(data, label) for(data, label) in zip(labels, data) if(label in classes)]
    filtered = list(filter(lambda x: x[0] in classes, zip(labels, data)))
    labels, data = zip(*filtered)

    return np.array(list(data)), np.array(list(labels))

def normalize_class(val, classes):
    return classes.index(val)

def normalize_classes(values, classes):
    return [normalize_class(v,classes) for v in values]

def my_load_data(classes = [0, 1,2,3,4,5,6,7,8,9], label_mode='fine'):
    """Loads CIFAR100 dataset.
    # Arguments
        label_mode: one of "fine", "coarse".
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    fpath = os.path.join(path, 'train')
    x_train, y_train_labels = my_load_batch(fpath, classes, label_key=label_mode + '_labels')
    y_train = normalize_classes(y_train_labels, classes)

    fpath = os.path.join(path, 'test')
    x_test, y_test_labels = my_load_batch(fpath, classes, label_key=label_mode + '_labels')
    y_test = normalize_classes(y_test_labels, classes)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_test /= 255
    x_train /= 255

    return (x_train, y_train), (x_test, y_test)