from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # Use this value as reference to calculate cumulative time taken
        self.start_time = tf.timestamp()
        self.train_start_time = None
        self.epoch_start_time = None
        self.test_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time_taken = tf.timestamp() - self.epoch_start_time
        self.times.append(epoch_time_taken)
        self.epochs.append(epoch)
        print(f"\nTime taken by epoch {epoch}: {epoch_time_taken:.2f} seconds")

    def on_test_begin(self, logs=None):
        self.test_start_time = tf.timestamp()

    def on_test_end(self, logs=None):
        test_end_time = tf.timestamp() - self.train_start_time
        print(f"\nTesting took {test_end_time:.2f} seconds")

    def on_train_begin(self, logs=None):
        self.train_start_time = tf.timestamp()

    def on_train_end(self, logs=None):
        train_end_time = tf.timestamp() - self.train_start_time
        print(f"\nTraining took {train_end_time:.2f} seconds")
        self.plot_time_taken(logs)

    def plot_time_taken(self, logs=None):
        plt.xlabel("Epoch")
        plt.ylabel("Time taken per epoch (seconds)")
        plt.plot(self.epochs, self.times, 'ro')
        for i in range(len(self.epochs)):
            j = self.times[i].numpy()
            if i == 0:
                plt.text(i, j, str(round(j, 3)))
            else:
                j_prev = self.times[i - 1].numpy()
                plt.text(i, j, str(round(j - j_prev, 3)))
        plt.savefig(datetime.now().strftime("TimeTaken-%Y_%m_%d_%H_%M_%S") + ".png")


class EarlyStoppingDesiredAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Early Stopping at desired accuracy, default 95%, callback.
    """

    def __init__(self, desired_accuracy=0.95):
        self.desired_accuracy = desired_accuracy

    def on_epoch_end(self, epoch, logs={}):
        if logs.get is not None and logs.get('accuracy'):
            desired_acc_pct = self.desired_accuracy * 100
            print(f"\nReached {desired_acc_pct}% accuracy after {epoch} epochs, so cancelling training!\n")
            self.model.stop_training = True


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"Up to batch {batch}, the average loss is {logs['loss']:7.2f}.")

    def on_test_batch_end(self, batch, logs=None):
        print(f"Up to batch {batch}, the average loss is {logs['loss']:7.2f}.")

    def on_epoch_end(self, epoch, logs=None):
        print(f"The average loss for epoch {epoch} is {logs['loss']:7.2f} \
        and mean absolute error is {logs['mean_absolute_error']:7.2f}.")


class EarlyStoppingMinLossCallback(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Args:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super(EarlyStoppingMinLossCallback, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class TensorBoardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, dir_name, experiment_name):
        self.log_dir = dir_name + "/" + experiment_name + "/" \
                       + datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Saving TensorBoard log files to: {self.log_dir}")
        super(TensorBoardCallback, self).__init__(dir_name, experiment_name)

# callbacks = [TimeCallback(), InterruptionCallback()]
