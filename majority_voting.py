from __future__ import print_function
from myutils import my_file_name_model
from CIFARS.mycifar100 import my_load_data
from keras.utils import to_categorical
import sys
from tensorflow.keras.models import load_model
import argparse
from keras.datasets import cifar10
import os
import numpy as np

parameters_passed = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest='enginedir', default='engine')
args = parser.parse_args()
enginedir = args.enginedir
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)



def choose_best(arr):
    arr = arr[0]
    j = 0
    max = arr[0]
    for i, e in enumerate(arr):
        if(e > max):
            max = e
            j = i
    return j

def load_engine():
    models = []
    names = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(enginedir):
        for file in f:
            if '.h5' in file:
                file_name = os.path.join(r, file)
                model = load_model(file_name)
                print('>loaded %s' % file_name)
                models.append(model)
                names.append(file_name)
    return (names, models)


def choose_best_vote(arr):
    max = arr[0]
    j = 0
    for i, e in enumerate(arr):
        if(e > max):
            j = i
            max = e
    return j

(names, models) = load_engine()

testy_enc = to_categorical(y_test, 10)
records = zip(x_test, testy_enc)
total = 0

for i, rec in enumerate(x_test):
    t = np.expand_dims(rec, 0).copy()
    votes = [0] * num_classes
    for m in models:
        predicted = m.predict(t, batch_size = 1)
        predict_res = choose_best(predicted)
        votes[predict_res - 1] += 1


    voted = choose_best_vote(votes) + 1
    if(voted == 10):
        voted = 0
    # print("voted is " + str(voted) + " and res is " + str(y_test[i]))
    if(voted == y_test[i]):
        total += 1
print('accuracy:', str(total) + "/" + str(len(x_test)))


