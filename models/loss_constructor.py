import os
import math
import pandas as pd
import tensorflow as tf
import numpy as np
from constants import NUM_CLASSES, TRAINING_DATA_PATH
import tensorflow as tf
import keras.backend as K


class Semantic_loss_functions(object):
    def __init__(self, weights_enabled=True):
        self.weights = np.ones((NUM_CLASSES,), dtype=np.float32)
        if weights_enabled:
            self.load_weights()
        print("semantic loss functions initialized")

    def load_weights(self):
        weights = pd.read_csv(
            os.path.join(TRAINING_DATA_PATH, "distribution.csv"), header=None
        )
        n = len(os.listdir(os.path.join(TRAINING_DATA_PATH, "x", "img")))
        for i in range(NUM_CLASSES):
            # tf-idf like calculation
            # self.weights[i] = math.log10(weights.iloc[i, 1]) * math.log10(n/weights.iloc[i, 2])
            self.weights[i] = math.log(weights.iloc[i, 1])
        self.weights = 1 - tf.nn.softmax(self.weights)
        print(self.weights)

    @tf.function
    def categorical_focal_loss(self, y_true, y_pred):
        gamma = 1.5
        alpha = 0.25
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=1)
        smooth = 1e-3
        loss = loss * smooth
        return loss

    @tf.function
    def categorical_jackard_loss(self, y_true, y_pred):
        """
        Jackard loss to minimize. Pass to model as loss during compile statement
        """

        intersection = K.sum(K.abs(y_true * y_pred), axis=-2)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-2)
        jac = (intersection) / (sum_ - intersection)

        return 1 - jac

    @tf.function
    def categorical_ssim_loss(self, y_true, y_pred, window_size=(4, 4)):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement
        """
        # calculate ssim for each channel seperately
        y_true = tf.reshape(y_true, [-1, window_size[0], 64, window_size[1], 64, 7])
        y_true = tf.transpose(y_true, perm=[0, 1, 3, 2, 4, 5])
        y_true = tf.reshape(y_true, [-1, window_size[0] * window_size[1], 64, 64, 7])

        y_pred = tf.reshape(y_pred, [-1, window_size[0], 64, window_size[1], 64, 7])
        y_pred = tf.transpose(y_pred, perm=[0, 1, 3, 2, 4, 5])
        y_pred = tf.reshape(y_pred, [-1, window_size[0] * window_size[1], 64, 64, 7])

        # sliding window ssim on separate channels
        categorical_ssim = tf.convert_to_tensor(
            [
                [
                    tf.image.ssim(
                        tf.expand_dims(y_true[:, j, :, :, i], -1),
                        tf.expand_dims(y_pred[:, j, :, :, i], -1),
                        max_val=1,
                        filter_size=25,
                        filter_sigma=2.5,
                        k1=0.06,
                        k2=0.02,
                    )
                    for i in range(7)
                ]
                for j in range(16)
            ]
        )
        # convert to loss
        categorical_ssim = 1 - categorical_ssim

        # calculate mean ssim for each channel
        tmp = tf.math.reduce_mean(categorical_ssim, axis=0)

        # calculate max ssim for each channel
        categorical_ssim = tf.math.reduce_max(categorical_ssim, axis=0)

        categorical_ssim = categorical_ssim + tmp
        # swap axes 0,1
        categorical_ssim = tf.transpose(categorical_ssim, perm=[1, 0])
        return categorical_ssim

    def ssim_loss(self, y_true, y_pred, window_size=(4, 4)):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement
        """
        # calculate ssim for each channel seperately
        y_true = tf.reshape(y_true, [-1, window_size[0], 64, window_size[1], 64, 7])
        y_true = tf.transpose(y_true, perm=[0, 1, 3, 2, 4, 5])
        y_true = tf.reshape(y_true, [-1, window_size[0] * window_size[1], 64, 64, 7])

        y_pred = tf.reshape(y_pred, [-1, window_size[0], 64, window_size[1], 64, 7])
        y_pred = tf.transpose(y_pred, perm=[0, 1, 3, 2, 4, 5])
        y_pred = tf.reshape(y_pred, [-1, window_size[0] * window_size[1], 64, 64, 7])

        # sliding window ssim on separate channels
        categorical_ssim = tf.convert_to_tensor(
            [
                [
                    tf.image.ssim_multiscale(
                        tf.expand_dims(y_true[:, j, :, :, i], -1),
                        tf.expand_dims(y_pred[:, j, :, :, i], -1),
                        max_val=1,
                        filter_size=2,
                    )
                    for i in range(7)
                ]
                for j in range(16)
            ]
        )
        # tf.print(categorical_ssim)
        categorical_ssim = tf.where(
            tf.math.is_nan(categorical_ssim), 0.0, categorical_ssim
        )
        # tf.print(categorical_ssim)
        # convert to loss
        categorical_ssim = 1 - categorical_ssim

        # calculate mean ssim for each channel
        tmp = tf.math.reduce_mean(categorical_ssim, axis=0)

        # calculate max ssim for each channel
        categorical_ssim = tf.math.reduce_max(categorical_ssim, axis=0)

        categorical_ssim = categorical_ssim + tmp

        # swap axes 0,1
        categorical_ssim = tf.transpose(categorical_ssim, perm=[1, 0])

        return categorical_ssim

    @tf.function
    def hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss to minimize. Pass to model as loss during compile statement.
        It is a combination of jackard loss, focal loss and ssim loss.
        """
        jackard_loss = self.categorical_jackard_loss(y_true, y_pred)
        focal_loss = self.categorical_focal_loss(y_true, y_pred)
        ssim_loss = self.categorical_ssim_loss(y_true, y_pred)

        jf = focal_loss + jackard_loss + ssim_loss

        # tf.print(type(jd))
        # tf.print(type(ssim_loss))
        return (jf) * self.weights
