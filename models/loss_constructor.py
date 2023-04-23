import os
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from constants import NUM_CLASSES, TRAINING_DATA_PATH
import tensorflow as tf
import keras.backend as K


class Semantic_loss_functions(object):
    def __init__(self, weights_enabled = True):
        self.weights = np.ones((NUM_CLASSES,),dtype=np.float32)
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
            self.weights[i] = math.log10(weights.iloc[i, 1])
        self.weights = 1 - tf.nn.softmax(self.weights)
        print(self.weights)



    def categorical_focal_loss(self, y_true, y_pred):
        gamma = 2.0
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

    def categorical_jackard_loss(self, y_true, y_pred):
        """
        Jackard loss to minimize. Pass to model as loss during compile statement
        """

        intersection = K.sum(K.abs(y_true * y_pred), axis=-2)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-2)
        jac = (intersection) / (sum_ - intersection)

        return 1 - jac

    def categorical_ssim_loss(self, y_true, y_pred):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement
        """
        # calculate ssim for each channel seperately
        y_true = tf.reshape(y_true, [-1, 256, 256, 7])
        y_pred = tf.reshape(y_pred, [-1, 256, 256, 7])

        categorical_ssim = tf.convert_to_tensor(
            [
                tf.image.ssim(
                    tf.expand_dims(y_true[..., i], -1),
                    tf.expand_dims(y_pred[..., i], -1),
                    max_val=1.0,
                    filter_size=11,
                )
                for i in range(7)
            ]
        )
        categorical_ssim = 1 - categorical_ssim

        # swap axes 0,1
        categorical_ssim = tf.transpose(categorical_ssim, perm=[1, 0])
        return categorical_ssim

    def ssim_loss(self, y_true, y_pred):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement
        """
        # calculate ssim for each channel seperately
        y_true = tf.reshape(y_true, [-1, 256, 256, 7])
        y_pred = tf.reshape(y_pred, [-1, 256, 256, 7])

        categorical_ssim = 1 - tf.reduce_mean(
            tf.image.ssim(
                y_true, y_pred, 1.0, filter_size=25, filter_sigma=2.5, k1=0.06, k2=0.02
            )
        )

        # swap axes 0,1

        return categorical_ssim

    def hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss to minimize. Pass to model as loss during compile statement.
        It is a combination of jackard loss, focal loss and ssim loss.
        """
        jackard_loss = self.categorical_jackard_loss(y_true, y_pred)
        focal_loss = self.categorical_focal_loss(y_true, y_pred)
        ssim_loss = self.categorical_ssim_loss(y_true, y_pred)

        return (jackard_loss * 1.3 + focal_loss + ssim_loss * 0.5) * self.weights
