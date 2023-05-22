import tensorflow as tf
import keras
from keras import backend as K
import time
import tensorflow as tf

class accuracy_drop_callback(keras.callbacks.Callback):
    previous_loss = None
    def on_epoch_end(self, epoch, logs={}):
        if not self.previous_loss is None and logs.get('loss') > self.previous_loss:
            #print("Loss has gotten worse. Reducing learning rate.")
            pass
            # learning_rate = self.model.optimizer.learning_rate
            # new_learning_rate = learning_rate * 0.1
            # tf.print(f"Reducing learning rate as loss has gotten worse. Decreasing from {learning_rate} to {new_learning_rate}")
            # self.learning_rate = new_learning_rate
            # self.model.compile(loss = self.model.loss_fn, optimizer = self.model.optimizer, metrics = ["accuracy"])
        else:
            #print("Loss has gotten better. Keeping learning rate.")
            pass
            # self.previous_loss = logs.get('loss')
            # new_learning_rate = learning_rate = self.model.optimizer.learning_rate
            # tf.print(f"Reducing learning rate as loss has gotten worse. Decreasing from {learning_rate} to {new_learning_rate}")
            # self.learning_rate = new_learning_rate
            # self.model.compile(loss = self.model.loss_fn, optimizer = self.model.optimizer, metrics = ["accuracy"])


class CustomReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    block_counter = 0
    def __init__(self, *args, **kwargs):
        super(CustomReduceLROnPlateau,self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        old_lr = float(K.get_value(self.model.optimizer.learning_rate))
        super(CustomReduceLROnPlateau, self).on_epoch_end(epoch, logs)
        new_lr = float(K.get_value(self.model.optimizer.learning_rate))
        block4_names = ["block4_conv1", "block4_conv2", "block4_conv3"]
        block3_names = ["block3_conv1", "block3_conv2", "block3_conv3"]
        if old_lr != new_lr:
            tf.print(f"Reducing learning rate as loss has gotten worse. Decreasing from {old_lr} to {new_lr}")

            if self.block_counter == 0:
                for layer in self.model.layers:
                    if layer.name in block4_names:
                        layer.trainable = True

            elif self.block_counter == 1:
                for layer in self.model.layers:
                    if layer.name in block3_names:
                        layer.trainable = True

            elif self.block_counter == 4:
                self.model.training = False

            tf.print(self.model.summary())
            self.block_counter += 1
            self.model.compile(loss = self.model.loss_fn, optimizer = self.model.optimizer, metrics = ["accuracy"])




