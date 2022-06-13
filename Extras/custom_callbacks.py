from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


class TimeAndPerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, show_text=True):
        self.time_started = None
        self.time_finished = None
        self.time_curr_epoch = None
        self.num_epochs = 0
        self.epoch_times = []
        self.epochs = []
        self._loss, self._acc, self._val_loss, self._val_acc = [], [], [], []
        self.start_stop_sep = "".join(["*"]*100)
        self.show_text = show_text
        
    def _plot_time_taken(self):
        plt.rcParams["figure.figsize"] = (8,6)
        plt.suptitle("Training Time")
        plt.xlabel("Epoch")
        plt.ylabel("Time per epoch (seconds)")
        plt.plot(self.epochs, self.epoch_times, 'r')
        if self.show_text:
            for i, j in zip(self.epochs, self.epoch_times):
                plt.text(i, j, str(round(j, 3)))
        plt.show()
    
    def _plot_model_performance(self):
        plt.rcParams["figure.figsize"] = (7,10)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Model performance', size=20)
        
        ax1.plot(range(self.num_epochs), self._loss, label='Training loss')
        ax1.plot(range(self.num_epochs), self._val_loss, label='Validation loss')
        ax1.set_xlabel('Epoch', size=14)
        ax1.set_ylabel('Loss', size=14)
        ax1.legend()
        
        ax2.plot(range(self.num_epochs), self._acc, label='Training accuracy')
        ax2.plot(range(self.num_epochs), self._val_acc, label='Validation Accuracy')
        ax2.set_xlabel('Epoch', size=14)
        ax2.set_ylabel('Accuracy', size=14)
        ax2.legend()
        plt.show()
        
    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f'TRAINING STARTED | {self.time_started}\n')
        print(f"{self.start_stop_sep}\n{self.start_stop_sep}")
        
    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = str(self.time_finished - self.time_started)
        print(f"\n{self.start_stop_sep}\n{self.start_stop_sep}")
        print(f'TRAINING FINISHED | {self.time_finished} | Duration: {train_duration}')
        
        tl = f"Training loss:       {logs['loss']:.5f}"
        ta = f"Training accuracy:   {logs['accuracy']:.5f}"
        vl = f"Validation loss:     {logs['val_loss']:.5f}"
        va = f"Validation accuracy: {logs['val_accuracy']:.5f}"
        
        print('\n'.join([tl, vl, ta, va]), "\n")
        self._plot_time_taken()
        self._plot_model_performance()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.time_curr_epoch = datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs += 1
        epoch_duration = (datetime.now() - self.time_curr_epoch).total_seconds()
        tl = logs['loss']
        ta = logs['accuracy']
        vl = logs['val_loss']
        va = logs['val_accuracy']
        
        self._loss.append(tl)
        self._acc.append(ta)
        self._val_loss.append(vl)
        self._val_acc.append(va)
        
        self.epoch_times.append(epoch_duration)
        self.epochs.append(epoch)

        train_metrics = f"train_loss: {tl:.5f}, train_accuracy: {ta:.5f}"
        valid_metrics = f"valid_loss: {vl:.5f}, valid_accuracy: {va:.5f}"
        
        print(f"Epoch: {epoch:4} | Runtime: {epoch_duration:.3f}s | {train_metrics} | {valid_metrics}")


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
    """
    Stop training when the loss is at its min, i.e. the loss stops decreasing.

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


class TensorBoardCallback(TensorBoard):
    """
    Create a TensorBoard callback.

    Unlike other custom callbacks, this callback extends the pre-build TensorBoard callback.
    """
    def __init__(self, dir_name, experiment_name):
        self.log_dir = dir_name + "/" + experiment_name + "/" \
                       + datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Saving TensorBoard log files to: {self.log_dir}")
        super().__init__(log_dir=self.log_dir)
