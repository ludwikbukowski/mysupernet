from __future__ import print_function
from myutils import my_file_name_model
from CIFARS.mycifar100 import my_load_data
from tensorflow.keras.utils import to_categorical
import sys
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.datasets import cifar10
import os

parameters_passed = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest='enginedir', default='engine')
args = parser.parse_args()
enginedir = args.enginedir
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def load_engine():
    models = []
    names = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(enginedir):
        for file in f:
            if '.hdf5' in file:
                file_name = os.path.join(r, file)
                model = load_model(file_name)
                print('>loaded %s' % file_name)
                models.append(model)
                names.append(file_name)
    return (names, models)


(names, models) = load_engine()
for n,m in zip(names, models):
    print("------model: " + n)
    testy_enc = to_categorical(y_test, 10)
    scores = m.evaluate(x_test, testy_enc, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


