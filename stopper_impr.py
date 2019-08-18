from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

from keras.datasets import cifar10
from myutils import my_file_name_model
from myutils import my_file_name_weights
from myutils import my_load_model
from myutils import random_name_gen
from myutils import save_experiment
from myutils import supernet_shuffle_and_reduce
from myutils import gen_save_dir
import numpy as np
import tensorflow as tf
from datetime import date
import sys
import argparse
from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler
from terminator import TerminateOnBaseline


loc = 'results/normal'
save_dir = gen_save_dir(loc)
parameters_passed = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', help="numer of epochs to iterate", type= int, default=100)
parser.add_argument('--batch-size', '-b', help="batch size", type= int, default=32)
parser.add_argument('--prefix', '-p', help="name for directory with results", type=str, default='model')
parser.add_argument('--stopat', '-s', help="stop when accuracy achived", type=float, default=0.9)
parser.add_argument('--notes', '-n', help="additional notes included in final raport", type=str, default='no notes')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
data_augmentation = True
notes = args.notes
stopat = args.stopat
num_classes = 10
model_name = my_file_name_model()

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

(x_train, y_train, _, _) = supernet_shuffle_and_reduce(x_train, y_train)


from keras import regularizers
weight_decay = 1e-4
model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


from keras.utils import plot_model
# initiate RMSprop optimizer

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

model.summary()
history = []

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    filepath = save_dir+"/saved-model-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=False)
    csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
    lr_sched = LearningRateScheduler(lr_schedule)
    terminator = TerminateOnBaseline(monitor='val_acc', baseline=stopat)
    history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=64),
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=(x_test, y_test),
                        workers=4,
                        verbose = 2,
                        callbacks=[csv_logger, checkpoint, terminator, lr_sched])


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


save_experiment(parameters_passed,save_dir, history, model, epochs, batch_size, reduce, scores, notes)
