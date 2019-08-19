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
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--branch', '-d', help="branch from root", type= int, default=1)
parser.add_argument('--trainroot', '-r', help="train root net", type= int, default=1)
parser.add_argument('--trainsubs', '-s', help="train subs", type= int, default=1)
parser.add_argument('--trainsuper', '-t', help="train subs", type= int, default=1)
parser.add_argument('--trainsuperall', '-f', help="train subs", type= int, default=0)
parser.add_argument('--epochs1', '-e', help="numer of epochs to iterate", type= int, default=20)
parser.add_argument('--epochs2', '-q', help="numer of epochs to iterate", type= int, default=20)
parser.add_argument('--epochs3', '-w', help="numer of epochs to iterate", type= int, default=20)
parser.add_argument('--epochs4', '-c', help="numer of epochs to iterate", type= int, default=20)
parser.add_argument('--alldata', '-a', help="numer of epochs to iterate", type= int, default=1)
parser.add_argument('--stopat1', '-g', help="numer of epochs to iterate", type= float, default=0.95)
parser.add_argument('--stopat2', '-l', help="numer of epochs to iterate", type= float, default=0.95)
parser.add_argument('--stopat3', '-j', help="numer of epochs to iterate", type= float, default=0.98)
parser.add_argument('--n', '-n', help="numer of submodels?", type= int, default=3)
parser.add_argument('--batch_size', '-b', help="batch_size", type= int, default=32)

args = parser.parse_args()
parameters_passed = sys.argv[1:]
branch = args.branch
trainroot = args.trainroot
trainsubs = args.trainsubs
trainsuper = args.trainsuper
trainsuperall = args.trainsuperall
epochs1 = args.epochs1
epochs2 = args.epochs2
epochs3 = args.epochs3
epochs4 = args.epochs4
alldata = args.alldata
stopat1 = args.stopat1
stopat2 = args.stopat2
stopat3 = args.stopat3
n_subs = args.n

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


print(str(input_dim))
model = Sequential()
model.add(Dense(300, input_dim=input_dim, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(3000  , activation =     "relu"))
model.add(Dropout(0.25))
model.add(Dense(3000  , activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation = "softmax"))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()

ter1 = TerminateOnBaseline(monitor='val_acc', baseline=stopat1)
filepath1 = save_dir + "/saved-model_fminst-root-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint1 = keras.callbacks.ModelCheckpoint(filepath1, monitor='val_acc', verbose=2, save_best_only=False)

saved = clone_model(model)
saved.set_weights(model.get_weights())

if(trainroot != 0):
    print("-------------------------------")
    print("Training Root...")

    csv_logger = CSVLogger(save_dir + '/history_root.csv', append=True, separator=';')
    callbacks = [ter1,csv_logger]
    if(branch!=0):
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='fminst_{epoch:02d}_tmp.h5', save_best_only=False,
                                                     monitor='val_acc', mode='max',
                                                     period=epochs1, verbose=1)
        callbacks.append(checkpoint)
    hist = model.fit(x_train, y_train, nb_epoch=epochs4, batch_size=batch_size ,validation_data = (x_test, y_test), verbose=2, callbacks = callbacks)
    print("Root trained.")
    model.summary()
    # final_plot(hist, "tmp.png")


# print("-------Evaluation base model")
# scores = model.evaluate(x_test, y_test, verbose=2)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])



def define_submodel(member, total, index, opt):
    submodel = Sequential()

    ## First layer and last is treated specially
    first_weights, first_bias = member.layers[0].get_weights()

    ((_, inp), (_, out)) = member.layers[0].input_shape, member.layers[0].output_shape
    part = int(out / total)
    new_weights = np.array([])
    for w in first_weights:
        start_at = index * part
        end_at = start_at + part
        new_w = np.array(w[start_at:end_at])
        new_weights = np.append(new_weights, new_w)
    new_weights = new_weights.reshape((inp, part))
    densefirst = Dense(part, activation=member.layers[0].activation, input_dim=inp, weights=[new_weights, first_bias[0:part]])
    submodel.add(densefirst)
    submodel.add(Dropout(0.25))

    ## hidden layers
    # model.summary()
    for l in model.layers[1:-1]:
        ((_, inp), (_, out)) = l.input_shape, l.output_shape
        part1 = int(inp / total)
        part2 = int(out / total)
        start_at1 = index * part1
        end_at1 = start_at1 + part1

        start_at2 = index * part2
        end_at2 = start_at2 + part2

        layer_type = type(l).__name__
        if (layer_type == 'Conv2D' or layer_type == 'Dense'):
            new_weights = np.array([])
            weights, bias = l.get_weights()
            for w in weights[start_at1:end_at1]:
                new_w = np.array(w[start_at2:end_at2])
                new_weights = np.append(new_weights, new_w)
            new_weights = new_weights.reshape((part1, part2))
            dense = Dense(part2, activation=l.activation,weights=[new_weights, bias[start_at2:end_at2]])
            submodel.add(dense)
            submodel.add(Dropout(0.25))

    ## last layer
    last_weights, last_bias = member.layers[-1].get_weights()
    ((_, inp), (_, out)) = member.layers[-1].input_shape, member.layers[-1].output_shape
    part1 = int(inp / total)

    start_at = index * part1
    end_at = start_at + part1
    new_weights = np.array([])
    for w in last_weights[start_at:end_at]:
        new_w = np.array(w)
        new_weights = np.append(new_weights, new_w)

    new_weights = new_weights.reshape((part1, out))
    lastlayer = Dense(nb_classes, activation=member.layers[-1].activation, weights=[new_weights, last_bias])
    submodel.add(lastlayer)
    submodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    submodel.summary()
    return submodel

if(trainroot!=0):
        if(branch!=0):
            saved = load_model("fminst_" + str(epochs1)+"_tmp.h5")

if(trainsubs!=0):
    subs = []
    # opts     = ['rmsprop', 'adam', 'adagrad', 'sgd', 'nadam', 'adadelta']
    for i,s in enumerate(range(n_subs)):
        sub = define_submodel(saved, n_subs, s, 'rmsprop')
        subs.append(sub)

# plot_model(sub0, show_shapes=True, to_file='sub0.png')

if(trainsubs !=0):
    print("-------------------------------")
    print("Training submodels...")
    ter2 = TerminateOnBaseline(monitor='val_acc', baseline=stopat2)
    for i,s in enumerate(subs):
        print("-------------------------------")
        filepath_x = save_dir + "/saved-model_fminst-sub" + str(i) + "-{epoch:02d}-{val_acc:.2f}.h5"
        checkpoint_x = keras.callbacks.ModelCheckpoint(filepath_x, monitor='val_acc', verbose=2, save_best_only=False)
        csv_logger_x = CSVLogger(save_dir + '/history_root_sub' + str(i) + '.csv', append=True, separator=';')
        s.fit(x_train, y_train, nb_epoch=epochs2, batch_size=batch_size ,validation_data = (x_test, y_test), verbose=2,
              initial_epoch=epochs1,
              callbacks = [ter2, csv_logger_x])
    print("Submodels trained.")

# print("-------Evaluation base model")
# scores = model.evaluate(x_test, y_test, verbose=2)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

if(trainsubs !=0):
    print("----------Evaluation subs model")
    for s in subs:
        scores = s.evaluate(x_test, y_test, verbose=2)
        # print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])


class Predictor:

    def __init__(self, model):
        self._model = model
        self._first_layer = model.get_layer(index=0).input

    def predict(self, x, index=-1):
        first_layer = self._first_layer
        last_layer = self._model.get_layer(index=index).output
        print(last_layer.name)
        new_model = Model(inputs=[first_layer], outputs=[last_layer])
        return new_model.predict(x)


def create_sets(members, x_train, X_test_my):
    predictors = tuple(Predictor(model) for model in members)
    if(trainsuper!=0):
        new_training_set = np.concatenate([predictor.predict(x_train, -2) for predictor in predictors], axis=1)
    else:
        new_training_set = np.array([])
    new_testing_set = np.concatenate([predictor.predict(X_test_my, -2) for predictor in predictors], axis=1)
    return (new_training_set, new_testing_set)

def define_supernet(members, new_training_set):
    model = Sequential([
        Dense(nb_classes, activation='softmax', input_shape=(new_training_set.shape[1],))
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    all_weights = [m.get_layer(index=-1).get_weights() for m in members]

    weights = np.concatenate([w for w, _ in all_weights], axis=0)
    biases = np.mean([b for _, b in all_weights], axis=0)

    model.get_layer(index=-1).set_weights([weights, biases])
    return model


def define_stacked_model(members, final_class_num):
  print(members)
  last_layer_weights = np.array([], ndmin=2)
  biases = []
  dims = 0
  for i in range(len(members)):
      model = members[i]
      model.name = '_' + str(i+1) + '_' + model.name
      for layer in model._layers:
          layer.trainable = True
          layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
      lastlayer = model._layers.pop()
      weights = lastlayer.get_weights()[0]
      (dim_val, final_class_num) = weights.shape
      dims += dim_val
      last_layer_weights = np.append(last_layer_weights, weights)
      bias = lastlayer.get_weights()[1]
      biases.append(bias)
  last_layer_weights = last_layer_weights.reshape((dims, final_class_num))
  ensemble_visible = [model.input for model in members]
  ensemble_outputs = [model.layers[-1].output for model in members]
  merge = concatenate(ensemble_outputs)
  output = Dense(final_class_num, activation='softmax'
                 )(merge)
  # last_layer_weights
  model = Model(inputs=ensemble_visible,  outputs=output)
  model._layers[-1].set_weights([last_layer_weights, np.mean(biases, axis = 0)])
  # below ignored for now
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  model.summary()

  return model



if(trainsuper!=0):
    print("-------------------------------")
    print("Train supermodel....")
    x_train_new, x_testing_new = create_sets(subs, x_train, x_test)
    supernet = define_supernet(subs, x_testing_new)
    # supernet.summary()
    ter3 = TerminateOnBaseline(monitor='val_acc', baseline=stopat3)
    csv_logger = CSVLogger(save_dir + '/history_super.csv', append=True, separator=';')
    supernet.fit(x_train_new, y_train,
              batch_size=int(batch_size),
                 initial_epoch=epochs2,
          epochs=epochs3,
          verbose=2,
          validation_data=(x_testing_new, y_test),
          callbacks=[ter3,csv_logger])
    supernet.summary()
    print("SuperModel trained")

    # scores = supernet.evaluate(x_testing_new, y_test, verbose=2)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

if(trainsuperall!=0):
    supernet = define_stacked_model(subs, 10)
    x_test = [x_test for _ in range(len(supernet.input))]
    x_train = [x_train for _ in range(len(supernet.input))]
    csv_logger = CSVLogger(save_dir + '/history_super.csv', append=True, separator=';')
    scores = supernet.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # supernet.fit(x_train, y_train,
    #              batch_size=int(batch_size),
    #              initial_epoch=epochs2,
    #              epochs=epochs3,
    #              verbose=2,
    #              validation_data=(x_test, y_test),
    #              callbacks=[csv_logger])
    print("SuperModel All trained")

