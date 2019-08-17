from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import concatenate
from tensorflow.keras.utils import plot_model
from numpy import argmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import cifar10
import keras
import numpy as np
from keras.layers.merge import concatenate
from myutils import my_load_model
from myutils import shuffle_and_reduce
from myutils import save_supernet
from myutils import gen_save_dir
from tensorflow.keras.utils import to_categorical
from datetime import date
from tensorflow.keras import regularizers
from myutils import final_plot
import os
import argparse
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger

loc = 'results/supernets'
save_dir = gen_save_dir(loc)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', help="numer of epochs to iterate", type= int, default=100)
parser.add_argument('--batch-size', '-b', help="batch size", type= int, default=32)
parser.add_argument('--reduce', '-r', help="how many training data is taken", type=float, default=1)
parser.add_argument('--datadog', '-d', help="how metrics will be named", type=str, default='supernet')
parser.add_argument('--notes', '-n', help="additional notes included in final raport", type=str, default='no notes')
parser.add_argument('--trainable', dest='trainable', action='store_true')
parser.add_argument('--no-trainable', dest='trainable', action='store_false')
parser.add_argument('--dir', dest='enginedir', default='engine')
parser.add_argument('--opt', dest='opt', default='rmsprop')
parser.set_defaults(trainable=False)

num_classes = 10
args = parser.parse_args()
parameters_passed = sys.argv[1:]
reduce_percent = args.reduce
batch_size = args.batch_size
epochs = args.epochs
datadogname = args.datadog
notes = args.notes
trainable = args.trainable
enginedir = args.enginedir
opt = args.opt

print('------------')
print('batch_size: ' + str(batch_size))
print('epochs: ' + str(epochs))
print('trainable: ' + str(trainable))
print('notes: ' + str(notes))
print('opt: ' + str(opt))
print('data used: ' + str(reduce_percent * 100) + '%')
print('------------')
if(opt=="adam"):
    opt = "adam"
elif(opt=="adagrad"):
    opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
elif(opt=="rmsprop"):
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
elif(opt=="nadam"):
    opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
elif(opt=="sgd"):
    opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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


def copy_input(model, name):
  return Input(shape=model.input_shape, name = name)

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


def create_sets(members, x_train, x_test):
    predictors = tuple(Predictor(model) for model in members)
    new_training_set = np.concatenate([predictor.predict(x_train, -2) for predictor in predictors], axis=1)
    new_testing_set = np.concatenate([predictor.predict(x_test, -2) for predictor in predictors], axis=1)
    return (new_training_set, new_testing_set)

def define_supernet(members, new_training_set):
    model = Sequential([
        Dense(num_classes, activation='softmax', input_shape=(new_training_set.shape[1],))
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    all_weights = [m.get_layer(index=-1).get_weights() for m in members]

    weights = np.concatenate([w for w, _ in all_weights], axis=0)
    biases = np.mean([b for _, b in all_weights], axis=0)

    model.get_layer(index=-1).set_weights([weights, biases])
    return model





(names, members) = load_engine()
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

x_train_new, x_test_new = create_sets(members, x_train, x_test)
supernet = define_supernet(members, x_train_new)


scores = supernet.evaluate(x_test_new, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
supernet.fit(x_train_new, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test_new, y_test),
          callbacks=[csv_logger])

scores = supernet.evaluate(x_test_new, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
