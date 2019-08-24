from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Model
from keras.layers import Input
from terminator import TerminateOnBaseline
from keras.datasets import fashion_mnist
import keras
import os
import argparse
import sys
import pandas as pd
from keras.models import clone_model
from keras.models import load_model
from keras.layers.merge import concatenate
from myutils import final_plot
import random
import numpy as np
from keras.callbacks import CSVLogger
from myutils import gen_save_dir
from keras.regularizers import l2
from custom_csv_logger import CustomCSVLogger
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--epochs1', '-e', help="numer of epochs to iterate", type= int, default=20)
parser.add_argument('--alldata', '-a', help="numer of epochs to iterate", type= int, default=1)
parser.add_argument('--stopat1', '-g', help="numer of epochs to iterate", type= float, default=0.95)
parser.add_argument('--batch_size', '-b', help="batch_size", type= int, default=32)
parser.add_argument('--dir', '-f', help="batch_size", type= str, default="engine")

args = parser.parse_args()
parameters_passed = sys.argv[1:]
epochs1 = args.epochs1
alldata = args.alldata
stopat1 = args.stopat1
dir = args.dir
curr_epoch = 0

batch_size = args.batch_size

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# pre-processing: divide by max and substract mean
input_dim = x_train.shape[1]
nb_classes = 10

# convert list of labels to binary class matrix
reduce_percent = 0.8
loc = 'results/normal'
save_dir = gen_save_dir(loc)
# part_percentage = round(reduce_percent * len(X_train))

root_part = 0.7
subs_part = 0.2
supernet_part = 0.1
## 255 600 600
# 148 348 348
from keras import regularizers


def train_chosen(base, x_train, x_test, y_train, y_test, index, curr_epoch):
    print("Training layer " + str(index) + "...")
    trainable_indexes = []
    conv_indexes = []
    for i in range(1, len(base._layers)):
        if base._layers[-i].count_params() > 0:
            layer_type = type(base._layers[-i]).__name__
            if(layer_type == 'Dense'):
                conv_indexes.append(-i)
            trainable_indexes.append(-i)

    indexed_layer = conv_indexes[index-1]
    x_train_new2, x_test_new2 = create_sets(base, x_train, x_test, indexed_layer)
    totrain = define_chosen(base, index, conv_indexes, trainable_indexes)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    totrain.fit(x_train_new2, y_train,
                 batch_size=batch_size,
                 epochs=curr_epoch+epochs1,
                 verbose=2,
                 initial_epoch=curr_epoch,
                 validation_data=(x_test_new2, y_test),
                 callbacks=[csv_logger])

    curr_epoch = epochs1 + curr_epoch

    print("Copying trained weights...")
    for i in trainable_indexes[:trainable_indexes.index(indexed_layer)+1]:
        # print(i)
        base._layers[i].set_weights(totrain._layers[i].get_weights())
        base._layers[i].trainable = True

    base.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    print("Trained layer: -" + str(index))
    return base


## e.g index = 3
def define_chosen(oldmodel, index, conv_indexes, trinable_indexes):
    newmodel = clone_model(oldmodel)
    newmodel.set_weights(oldmodel.get_weights())


    indexed_layer = conv_indexes[index-1]
    print(indexed_layer)


    layers = newmodel.layers[indexed_layer:]
    print(layers)
    model = Sequential(layers)

    if(indexed_layer == -len(oldmodel.layers)):
        model = oldmodel
    else:
        model.build(newmodel.layers[indexed_layer-1].output_shape)
    # model.summary()

    last_nont_trainable = conv_indexes[index-1]

    for i in trinable_indexes[:trinable_indexes.index(last_nont_trainable) ]:
        model._layers[i].trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    print("Setting weights")
    for i in trinable_indexes[:trinable_indexes.index(indexed_layer)+1]:
        model._layers[i].set_weights(oldmodel.layers[i].get_weights())


    return model


class Predictor:

    def __init__(self, model):
        self._model = model
        self._first_layer = model.get_layer(index=0).input

    def predict(self, x, index=-1):
        first_layer = self._first_layer
        last_layer = self._model.get_layer(index=index).output

        new_model = Model(inputs=[first_layer], outputs=[last_layer])
        return new_model.predict(x)


def create_sets(model, x_train, x_test, index):
    print("Creating dataset foor layer " + str(index) + "...")
    predictor = Predictor(model)
    if(index == -len(model.layers)):
        return x_train, x_test
    new_training_set = predictor.predict(x_train, index-1)
    new_testing_set = predictor.predict(x_test, index-1)
    print("Dataset created.")
    return (new_training_set, new_testing_set)


model = Sequential()
model.add(Dense(300 , input_dim=input_dim, activation = "relu",))
# model.add(Dropout(0.2))
model.add(Dense(600 , activation = "relu",))
# model.add(Dropout(0.2))
model.add(Dense(600 ,  activation = "relu",))
# model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

base = clone_model(model)
base.set_weights(model.get_weights())
index=1

# print(base.layers[-1].get_weights())
# base = train_chosen(base, x_train, x_test, y_train, y_test, 2)
# print(base.layers[-1].get_weights())
c = 0
for _ in range(10):
    base = train_chosen(base, x_train, x_test, y_train, y_test, 1, c)
    c+= epochs1
    base = train_chosen(base, x_train, x_test, y_train, y_test, 2, c)
    c += epochs1
    base = train_chosen(base, x_train, x_test, y_train, y_test, 3, c)
    c += epochs1
    base = train_chosen(base, x_train, x_test, y_train, y_test, 4, c)
    c += epochs1
scores = base.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# base = train_chosen(base, x_train, x_test, y_train, y_test, 2)
