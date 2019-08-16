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
parser.add_argument('--epochs', '-e', help="numer of epochs to iterate", type= int, default=10)
parser.add_argument('--index', '-i', help="which layer to train starting from last", type= int, default=1)
parser.add_argument('--batch_size', help="batch size", type= int, default=32)
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
batch_size = args.batch_size
epochs = args.epochs
index = args.index
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
print('batch_size: ' + str(batch_size))
print('epochs1: ' + str(epochs))
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
    print("Creating dataset foor layer " + str(index) + "...")
    predictor = Predictor(model)
    new_training_set = predictor.predict(x_train, index-1)
    new_testing_set = predictor.predict(x_test, index-1)
    print("Dataset created.")
    return (new_training_set, new_testing_set)


from keras import backend as K
def get_layer_output_grad(model, inputs, outputs, layer=-1):

    model.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad



def train_chosen(model, base, x_train, x_test, y_train, y_test, index):
    print("Training layer " + str(index) + "...")
    trainable_indexes = []
    conv_indexes = []
    for i in range(1, len(base._layers)):
        if base._layers[-i].count_params() > 0:
            layer_type = type(base._layers[-i]).__name__
            if(layer_type == 'Conv2D' or layer_type == 'Dense'):
                conv_indexes.append(-i)
            trainable_indexes.append(-i)

    indexed_layer = conv_indexes[index-1]
    x_train_new2, x_test_new2 = create_sets(model, x_train, x_test, indexed_layer)
    totrain = define_chosen(base, index, conv_indexes, trainable_indexes)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    totrain.fit(x_train_new2, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_data=(x_test_new2, y_test),
                 callbacks=[csv_logger])

    print("Copying trained weights...")
    for i in trainable_indexes[:trainable_indexes.index(indexed_layer)+1]:
        # print(i)
        base._layers[i].set_weights(totrain._layers[i].get_weights())

    base.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    print("Trained layer: -" + str(index))
    return base


## e.g index = 3
def define_chosen(oldmodel, index, conv_indexes, trinable_indexes):
    newmodel = clone_model(oldmodel)
    newmodel.set_weights(oldmodel.get_weights())


    indexed_layer = conv_indexes[index-1    ]


    layers = newmodel.layers[indexed_layer:]
    model = Sequential(layers)
    model.build(newmodel.layers[indexed_layer-1].output_shape)
    # model.summary()

    last_nont_trainable = conv_indexes[index-1]

    for i in trinable_indexes[:trinable_indexes.index(last_nont_trainable) ]:
        model._layers[i].trainable = trainable

    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    print("Setting weights")
    for i in trinable_indexes[:trinable_indexes.index(indexed_layer)+1]:
        model._layers[i].set_weights(oldmodel.layers[i].get_weights())

    return model

# def train_single_chosen(model, base, x_train, x_test, y_train, y_test, index):
#     print("Training layer " + str(index) + "...")
#     trainable_indexes = []
#     conv_indexes = []
#     for i in range(1, len(base._layers)):
#         if base._layers[-i].count_params() > 0:
#             layer_type = type(base._layers[-i]).__name__
#             if(layer_type == 'Conv2D' or layer_type == 'Dense'):
#                 conv_indexes.append(-i)
#             trainable_indexes.append(-i)
#
#     indexed_layer = conv_indexes[index-1]
#     x_train_new2, x_test_new2 = create_sets(model, x_train, x_test, indexed_layer)
#     totrain = define_single_chosen(base, index, conv_indexes, trainable_indexes)
#     csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
#     totrain.fit(x_train_new2, y_train,
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  validation_data=(x_test_new2, y_test),
#                  callbacks=[csv_logger])
#
#     print("Copying trained weights...")
#     # for i in trainable_indexes[:trainable_indexes.index(indexed_layer)+1]:
#     #     print(i)
#     base._layers[indexed_layer].set_weights(totrain._layers[0].get_weights())
#
#     base.compile(loss='categorical_crossentropy',
#                  optimizer="adam",
#                  metrics=['accuracy'])
#     print("Trained layer: -" + str(index))
#     return base

# def define_single_chosen(oldmodel, index, conv_indexes, trinable_indexes):
#     newmodel = clone_model(oldmodel)
#     newmodel.set_weights(oldmodel.get_weights())
#
#
#     indexed_layer = conv_indexes[index-1]
#
#
#     layers = newmodel.layers[indexed_layer]
#     model = Sequential([layers])
#     model.build(newmodel.layers[indexed_layer-1].output_shape)
#     # model.summary()
#
#     last_nont_trainable = conv_indexes[index-1]
#
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer="adam",
#                   metrics=['accuracy'])
#     model.summary()
#     oldmodel.summary()
#
#     print("Setting weights")
#     # for i in trinable_indexes[:trinable_indexes.index(indexed_layer)]:
#     model._layers[1].set_weights(oldmodel.layers[indexed_layer].get_weights())
#
#
#     return model


model = load_model(filename)
print('>loaded %s' % filename)

## save base model to restore later


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

(x_train, y_train) = shuffle_and_reduce(reduce_percent, x_train, y_train)
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

base = clone_model(model)
base.set_weights(model.get_weights())

# gradients_train = get_layer_output_grad(base, x_train, y_train, -6)
# gradients_test = get_layer_output_grad(base, x_test, y_test, -6)
# print(gradients)

base = train_chosen(model, base, x_train, x_test, y_train, y_test, index)

# scores = base.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

base.save(output)




