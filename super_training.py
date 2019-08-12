from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from keras import Sequential
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import concatenate
from tensorflow.keras.utils import plot_model
from numpy import argmax
from keras.layers import Dense
from keras.datasets import cifar10
import numpy as np
from keras.layers.merge import concatenate
from myutils import my_load_model
from myutils import shuffle_and_reduce
from myutils import save_supernet
from myutils import gen_save_dir
from keras.utils import to_categorical
from datetime import date
from tensorflow.keras import regularizers
from myutils import final_plot
import os
import argparse
import sys
from keras.models import load_model
from keras.models import clone_model
from keras.callbacks import CSVLogger

loc = 'results/supernets'
save_dir = gen_save_dir(loc)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs1', '-e', help="numer of epochs to iterate", type= int, default=10)
parser.add_argument('--epochs2', '-f', help="numer of epochs to iterate", type= int, default=10)
parser.add_argument('--epochs3', '-q', help="numer of epochs to iterate", type= int, default=10)
parser.add_argument('--batch-size1', help="batch size", type= int, default=32)
parser.add_argument('--batch-size2', help="batch size", type= int, default=32)
parser.add_argument('--batch-size3', help="batch size", type= int, default=32)
parser.add_argument('--reduce', '-r', help="how many training data is taken", type=float, default=1)
parser.add_argument('--datadog', '-d', help="how metrics will be named", type=str, default='supernet')
parser.add_argument('--notes', '-n', help="additional notes included in final raport", type=str, default='no notes')
parser.add_argument('--trainable', dest='trainable', action='store_true')
parser.add_argument('--nolast', dest='last', action='store_false')
parser.add_argument('--notwo', dest='two', action='store_false')
parser.add_argument('--noone', dest='one', action='store_false')
parser.add_argument('--no-trainable', dest='trainable', action='store_false')
parser.add_argument('--model', dest='enginedir', default='engine')
parser.add_argument('--output', dest='output', default='saved_optimized_model.h5')
parser.add_argument('--opt', dest='opt', default='rmsprop')
parser.set_defaults(trainable=True)
parser.set_defaults(last=True)
parser.set_defaults(one=True)
parser.set_defaults(two=True)

num_classes = 10
args = parser.parse_args()
parameters_passed = sys.argv[1:]
reduce_percent = args.reduce
batch_size1 = args.batch_size1
batch_size2 = args.batch_size2
batch_size3 = args.batch_size3
epochs1 = args.epochs1
epochs2 = args.epochs2
epochs3 = args.epochs3
datadogname = args.datadog
notes = args.notes
trainable = args.trainable
last = args.last
one = args.one
two = args.two
filename = args.enginedir
output = args.output
opt = args.opt

print('------------')
print('batch_size: ' + str(batch_size1))
print('epochs1: ' + str(epochs1))
print('epochs2: ' + str(epochs2))
print('trainable: ' + str(trainable))
print('notes: ' + str(notes))
print('opt: ' + str(opt))
print('data used: ' + str(reduce_percent * 100) + '%')
print('------------')


def copy_input(model, name):
  return Input(shape=model.input_shape, name = name)

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
    predictor = Predictor(model)
    new_training_set = predictor.predict(x_train, index-1)
    new_testing_set = predictor.predict(x_test, index-1)
    return (new_training_set, new_testing_set)

def define_last_training(oldmodel, new_training_set):
    model = Sequential([
        Dense(num_classes, activation='softmax', input_shape=(new_training_set.shape[1],))
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    all_weights = oldmodel.get_layer(index=-1).get_weights()


    model.get_layer(index=-1).set_weights(all_weights)
    return model


def define_one_but_last_training(oldmodel):
    newmodel = clone_model(oldmodel)
    newmodel.set_weights(oldmodel.get_weights())

    layers = newmodel.layers[-7:]
    model = Sequential(layers)
    model.build(newmodel.layers[-8].output_shape)
    # model.summary()
    model._layers[-1].trainable = trainable
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model._layers[-1].set_weights(oldmodel.layers[-1].get_weights())
    model._layers[-5].set_weights(oldmodel.layers[-5].get_weights())
    model._layers[-7].set_weights(oldmodel.layers[-7].get_weights())

    return model

def define_two_but_last_training(oldmodel):
    newmodel = clone_model(oldmodel)
    newmodel.set_weights(oldmodel.get_weights())

    layers = newmodel.layers[-10:]
    model = Sequential(layers)
    model.build(newmodel.layers[-11].output_shape)
    # model.summary()
    model._layers[-1].trainable = trainable
    model._layers[-5].trainable = trainable
    model._layers[-7].trainable = trainable
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model._layers[-1].set_weights(oldmodel.layers[-1].get_weights())
    model._layers[-5].set_weights(oldmodel.layers[-5].get_weights())
    model._layers[-7].set_weights(oldmodel.layers[-7].get_weights())
    model._layers[-8].set_weights(oldmodel.layers[-8].get_weights())
    model._layers[-10].set_weights(oldmodel.layers[-10].get_weights())

    return model


def last_layer(model, base, x_train, x_test, y_train, y_test):

    x_train_new, x_test_new = create_sets(model, x_train, x_test, -1)
    supernet = define_last_training(model, x_train_new)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    supernet.fit(x_train_new, y_train,
                 batch_size=batch_size1,
                 epochs=epochs1,
                 validation_data=(x_test_new, y_test),
                 callbacks=[csv_logger])

    base._layers[-1].set_weights(supernet._layers[-1].get_weights())

    base.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    print("Trained last layer")
    return base

## -1 -5 and -7 are the one but last trainable layers
def one_but_last(model, base, x_train, x_test, y_train, y_test):
    x_train_new2, x_test_new2 = create_sets(model, x_train, x_test, -7)
    supernet = define_one_but_last_training(base)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    supernet.fit(x_train_new2, y_train,
                 batch_size=batch_size2,
                 epochs=epochs2,
                 validation_data=(x_test_new2, y_test),
                 callbacks=[csv_logger])

    base._layers[-1].set_weights(supernet._layers[-1].get_weights())
    base._layers[-5].set_weights(supernet._layers[-5].get_weights())
    base._layers[-7].set_weights(supernet._layers[-7].get_weights())
    base.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    print("Trained one but last layer")
    return base

# from keras import backend as K
# def get_layer_output_grad(model, inputs, outputs, layer=-1):
#     """ Gets gradient a layer output for given inputs and outputs"""
#     grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad

def two_but_last(model, base, x_train_new2, x_test_new2, y_train, y_test):

    supernet = define_two_but_last_training(base)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    supernet.fit(x_train_new2, y_train,
                 batch_size=batch_size3,
                 epochs=epochs3,
                 validation_data=(x_test_new2, y_test),
                 callbacks=[csv_logger])

    base._layers[-1].set_weights(supernet._layers[-1].get_weights())
    base._layers[-5].set_weights(supernet._layers[-5].get_weights())
    base._layers[-7].set_weights(supernet._layers[-7].get_weights())
    base._layers[-8].set_weights(supernet._layers[-8].get_weights())
    base._layers[-10].set_weights(supernet._layers[-10].get_weights())
    base.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    print("Trained two but last layer")
    return base


model = load_model(filename)
print('>loaded %s' % filename)
# model.summary()
## save base model to restore later


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
(x_train, y_train) = shuffle_and_reduce(reduce_percent, x_train, y_train)
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# print("generating gradients")
# r = get_layer_output_grad(model, x_train, y_train)
# print(str(r))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

base = clone_model(model)
base.set_weights(model.get_weights())



x_train_new2, x_test_new2 = create_sets(model, x_train, x_test, -10)

if(two == True):
    base = two_but_last(model, base, x_train_new2, x_test_new2, y_train, y_test)

if(one == True):
    base = one_but_last(model, base, x_train, x_test, y_train, y_test)

if (last == True):
    base = last_layer(model, base, x_train, x_test, y_train, y_test)



# if(last == True):
#     base = last_layer(model, base, x_train, x_test, y_train, y_test)

base.save(output)






# x_train_new2, x_test_new2 = create_sets(model, x_train, x_test, -7)
# supernet = define_one_but_last_training(base)

# scores = supernet.evaluate(x_test_new2, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
#
# supernet.fit(x_train_new2, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(x_test_new2, y_test),
#           callbacks=[csv_logger])


# base._layers[-1].set_weights(supernet._layers[-1].get_weights())
# base._layers[-5].set_weights(supernet._layers[-5].get_weights())
# base._layers[-7].set_weights(supernet._layers[-7].get_weights())
# base.compile(loss='categorical_crossentropy',
#               optimizer="adam",
#               metrics=['accuracy'])


# base.save(output)



