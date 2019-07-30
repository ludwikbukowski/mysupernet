from __future__ import print_function
import functools
import operator
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import string
import datetime
import os
from keras.utils import plot_model

def save_supernet(params, parent_dir, history, model, epochs, batch_size, reduced, scores, names, notes):
    aggregate_extra = '\n'
    for n in names:
        aggregate_extra = aggregate_extra + ' ' + n + '\n'
    dir = save_experiment(params, parent_dir, history, model, epochs, batch_size, reduced, scores, aggregate_extra, notes)
    plot_model(model, show_shapes=True, to_file=dir+'/model_graph.png')

def gen_save_dir(parent_dir):
    save_dir = os.path.join(os.getcwd(), parent_dir)
    save_dir = os.path.join(save_dir, random_name_gen())
    os.mkdir(save_dir)
    print('Creating directory at %s ' % save_dir)
    return save_dir

def save_experiment(params, save_dir, history, model, epochs, batch_size, reduced, scores, extra = '', notes = ''):
    model_name = my_file_name_model()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    details_path = os.path.join(save_dir, 'details.txt')
    f = open(details_path, "w+")
    f.write('parameters: ' + str(params))
    f.write('epochs: ' + str(epochs))
    f.write('\nbatch_size: ' + str(batch_size))
    f.write('\ndata percent: ' + str(reduced * 100) + '%')
    f.write('\ntest loss: ' + str(scores[0]))
    f.write('\ntest acc: ' + str(scores[1]))
    f.write('\nnotes: ' + str(notes))
    f.write('\nextra: ' + str(extra))
    f.close()
    final_plot(history, os.path.join(save_dir, 'plot.png'))
    print('Saved data at %s ' % save_dir)
    return save_dir


def random_name_gen(prefix='model'):
    current = datetime.datetime.now()
    current_str = current.strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + '_' + current_str + id_generator(3)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))

# load models from file
def my_load_model(classes):
    filename = my_file_name_model(classes)
    model = load_model('goods/' + filename)
    print('>loaded %s' % filename)
    return model

def my_file_name_model():
    classes_suffix = '_'
    classes_suffix_name = functools.reduce(operator.add, classes_suffix, 'keras_cifar100_trained_model')
    return classes_suffix_name + '.h5'

def my_file_name_weights(classes):
    classes_suffix = ['_' + str(c) for c in classes]
    classes_suffix_name = functools.reduce(operator.add, classes_suffix, 'keras_cifar100_trained_weights')
    return classes_suffix_name + '.h5'

def final_plot(history, loc):
    myhistory_acc = []
    myhistory_val_acc = []
    myhistory_val_loss = []
    myhistory_loss = []
    if isinstance(history, (list,)):
        for l in history:
            myhistory_acc = sum(l.history['acc'], myhistory_acc)
            myhistory_val_acc = sum(l.history['val_acc'], myhistory_val_acc)
            myhistory_loss= sum(l.history['loss'], myhistory_loss)
            myhistory_val_loss= sum(l.history['val_loss'], myhistory_val_loss)
    else:
        myhistory_acc = history.history['acc']
        myhistory_val_acc = history.history['val_acc']
        myhistory_loss = history.history['loss']
        myhistory_val_loss = history.history['val_loss']

    # grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    fig = plt.figure()
    plt. subplots_adjust(hspace = 0.001)
    fig.suptitle('ludwik.bukowski@gmail.com', fontsize=10, fontweight='bold')
    ax = fig.add_subplot(211)
    fig.subplots_adjust(top=0.95)
    ax.set_title('accuracy', fontsize=10)
    ax.set_xlabel('epoch')
    # plt.grid(True, lw=2, ls='--', c='.75')
    ax.set_ylabel('acc')
    ax.plot(myhistory_acc, label="acc")
    ax.plot(myhistory_val_acc, label="val_acc")
    plt.legend(['train', 'test'], loc='upper left')

    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(top=0.85)
    ax2.set_title('loss', fontsize=10)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.plot(myhistory_loss, label="loss")
    ax2.plot(myhistory_val_loss, label="val_loss")
    plt.legend(['train', 'test'], loc='upper left')
    # plt.grid(True, lw=2, ls='--', c='.75')
    plt.savefig(loc)
    plt.show()

def shuffle_and_reduce(reduce_percent, x_data, y_data):
    # print("------- using " + str(reduce_percent * 100) + '% of data! --------')
    merged = list(zip(x_data, y_data))
    random.shuffle(merged)
    (x_data, y_data) = zip(*merged)
    x_data = list(x_data)
    y_data = list(y_data)
    part_percentage = round(reduce_percent * len(x_data))
    x_data = np.array(x_data[0:part_percentage])
    y_data = np.array(y_data[0:part_percentage])
    return (x_data, y_data)


def supernet_shuffle_and_reduce(x_data, y_data, reduce_percent = 0.7, ):
    # print("------- using " + str(reduce_percent * 100) + '% of data! --------')
    merged = list(zip(x_data, y_data))

    ## Dont shuffle on purpose
    # random.shuffle(merged)
    (x_data, y_data) = zip(*merged)
    x_data = list(x_data)
    y_data = list(y_data)
    part_percentage = round(reduce_percent * len(x_data))
    x_data_1 = np.array(x_data[0:part_percentage])
    y_data_1 = np.array(y_data[0:part_percentage])
    x_supernet_data = np.array(x_data[part_percentage: len(x_data)])
    y_supernet_data = np.array(y_data[part_percentage: len(y_data)])
    return (x_data_1, y_data_1, x_supernet_data, y_supernet_data)