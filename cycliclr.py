import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback


class MyCyclic(Callback):

    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and dir4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, steps_per_epoch, cycle = 4, min_lr=1e-2, max_lr=1e-5):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle = cycle
        self.total_iterations = steps_per_epoch
        self.iteration = 0
        self.epoch = 0
        self.current_lr = 0
        self.history = {}
        print("steps per epochs is " + str(steps_per_epoch))

    def schedule_lr(self, iter, epoch, cycle, alpha_1, alpha_2):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        mylr = self.schedule_lr(x, self.epoch, self.cycle, self.min_lr, self.max_lr)
        return mylr

    def on_train_begin(self, logs=None):
        logs = logs or {}

    def on_epoch_begin(self, epoch, logs=None):
        # print(" [EPOCH] Learning rate is " + str(self.current_lr))
        self.epoch += 1
        self.iteration = 1

    def on_epoch_end(self, epoch, logs=None):
        print("there was " + str(self.iteration) + " iterations")
        self.total_iterations = self.iteration

    def on_batch_begin(self, epoch, logs=None):
        newlr = self.clr()
        # print("[BATCH] Learning rate is " + str(newlr))
        self.current_lr = newlr
        K.set_value(self.model.optimizer.lr, newlr)
        self.iteration += 1

        # self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        # self.history.setdefault('iterations', []).append(self.iteration)
        #
        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)



    # def on_batch_end(self, epoch, logs=None):
    #     '''Record previous batch statistics and update the learning rate.'''
        # logs = logs or {}
        # self.iteration += 1
        #
        # self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        # self.history.setdefault('iterations', []).append(self.iteration)
        #
        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)
        #
        # K.set_value(self.model.optimizer.lr, self.clr())

    # def plot_loss(self):
    #     '''Helper function to quickly observe the learning rate experiment results.'''
    #     plt.plot(self.history['lr'], self.history['loss'])
    #     plt.xscale('log')
    #     plt.xlabel('Learning rate')
    #     plt.ylabel('Loss')
    #     plt.show()