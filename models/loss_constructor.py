import math
import os

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf


class SemanticLoss(object):
    """
    Class for semantic loss functions

    Attributes
    ----------
    :bool weights_enabled: whether to use class weights
    """

    def __init__(
        self,
        n_classes,
        weights_enabled=True,
        weights_path=None,
    ):
        
        self.n_classes = n_classes
        self.alpha = 0.25
        self.gamma = 1.5
        self.window_size = (4, 4)
        self.filter_size = 2
        self.filter_sigma = 1.5
        self.k1 = 0.01
        self.k2 = 0.03

        self.weights = np.ones((n_classes,), dtype=np.float32)
        if weights_enabled:
            self.load_weights(weights_path)
        print(f"Semantic loss function initialized with {n_classes} classes.")

    def set_alpha(self, alpha):
        """
        Set alpha parameter for focal loss

        Parameters
        ----------
        :float alpha: alpha parameter for focal loss
        """
        self.alpha = alpha

    def set_gamma(self, gamma):
        """
        Set gamma parameter for focal loss

        Parameters
        ----------
        :float gamma: gamma parameter for focal loss
        """
        self.gamma = gamma

    def set_weights(self, weights):
        """
        Set class weights

        Parameters
        ----------
        :list weights: list of class weights
        """
        self.weights = weights

    def set_window_size(self, window_size):
        """
        Set window size for ssim loss

        Parameters
        ----------
        :tuple window_size: window size for ssim loss
        """
        self.window_size = window_size

    def set_filter_size(self, filter_size):
        """
        Set filter size for ssim loss

        Parameters
        ----------
        :int filter_size: filter size for ssim loss
        """
        self.filter_size = filter_size

    def set_filter_sigma(self, filter_sigma):
        """
        Set filter sigma for ssim loss

        Parameters
        ----------
        :float filter_sigma: filter sigma for ssim loss
        """
        self.filter_sigma = filter_sigma

    def set_k1(self, k1):
        """
        Set k1 parameter for ssim loss

        Parameters
        ----------
        :float k1: k1 parameter for ssim loss
        """
        self.k1 = k1

    def set_k2(self, k2):
        """
        Set k2 parameter for ssim loss

        Parameters
        ----------
        :float k2: k2 parameter for ssim loss
        """
        self.k2 = k2

    def load_weights(self,path):
        """
        Load class weights from csv file
        """
        weights = pd.read_csv(
            os.path.join(path, "distribution.csv"), header=None
        )

        for i in range(self.n_classes):
            
            self.weights[i] = math.log2(weights.iloc[i, 1])
        self.weights = 1 - tf.nn.softmax(self.weights)

        join_str = ", "
        print(
            f"Class weights initialized as: {join_str.join([str(round(x,4)) for x in K.eval(self.weights)])}"
        )

    @tf.function
    def categorical_focal_loss(self, y_true, y_pred):
        """
        Calculate focal loss separate for each class

        Parameters
        ----------
        :tensor y_true: ground truth mask
        :tensor y_pred: predicted mask

        Returns
        -------
        :tensor loss: focal loss
        """
        gamma = self.gamma
        alpha = self.alpha
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=(1, 2))
        smooth = 1e-3
        loss = loss * smooth
        return loss

    @tf.function
    def categorical_jackard_loss(self, y_true, y_pred):
        """
        Jackard loss to minimize. Pass to model as loss during compile statement

        Parameters
        ----------
        :tensor y_true: ground truth mask
        :tensor y_pred: predicted mask

        Returns
        -------
        :tensor loss: jackard loss
        """

        intersection = K.sum(K.abs(y_true * y_pred), axis=(-3, -2))
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=(-3, -2))
        jac = (intersection) / (sum_ - intersection)

        return 1 - jac

    @tf.function
    def categorical_ssim_loss(self, y_true, y_pred):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement

        Parameters
        ----------
        :tensor y_true: ground truth mask
        :tensor y_pred: predicted mask

        Returns
        -------
        :tensor loss: ssim loss
        """
        window_size = self.window_size
        tile_size = y_true.shape[1] // window_size[0]
        # calculate ssim for each channel seperately
        y_true = tf.reshape(
            y_true, [-1, window_size[0], tile_size, window_size[1], tile_size, 7]
        )
        y_true = tf.transpose(y_true, perm=[0, 1, 3, 2, 4, 5])
        y_true = tf.reshape(
            y_true, [-1, window_size[0] * window_size[1], tile_size, tile_size, 7]
        )

        y_pred = tf.reshape(
            y_pred, [-1, window_size[0], tile_size, window_size[1], tile_size, 7]
        )
        y_pred = tf.transpose(y_pred, perm=[0, 1, 3, 2, 4, 5])
        y_pred = tf.reshape(
            y_pred, [-1, window_size[0] * window_size[1], tile_size, tile_size, 7]
        )

        # sliding window ssim on separate channels
        categorical_ssim = tf.convert_to_tensor(
            [
                [
                    tf.image.ssim(
                        tf.expand_dims(y_true[:, j, :, :, i], -1),
                        tf.expand_dims(y_pred[:, j, :, :, i], -1),
                        max_val=1,
                        filter_size=self.filter_size,
                        filter_sigma=self.filter_sigma,
                        k1=self.k1,
                        k2=self.k2,
                    )
                    for i in range(7)
                ]
                for j in range(16)
            ]
        )
        # convert to loss
        categorical_ssim = 1 - categorical_ssim

        # calculate mean ssim for each channel
        categorical_ssim = tf.math.reduce_sum(categorical_ssim, axis=0)

        # calculate max ssim for each channel
        # categorical_ssim = tf.math.reduce_max(categorical_ssim, axis=0)

        # categorical_ssim = categorical_ssim + tmp
        # swap axes 0,1
        categorical_ssim = tf.transpose(categorical_ssim, perm=[1, 0])
        return categorical_ssim

    @tf.function
    def ssim_loss(self, y_true, y_pred, window_size=(4, 4)):
        """
        SSIM loss to minimize. Pass to model as loss during compile statement

        Parameters
        ----------
        :tensor y_true: ground truth mask
        :tensor y_pred: predicted mask
        :tuple window_size: window size for ssim loss

        Returns
        -------
        :tensor loss: ssim loss
        """
        # calculate ssim for each channel seperately
        tile_size = y_true.shape[1] // window_size[0]

        y_true = tf.reshape(
            y_true, [-1, window_size[0], tile_size, window_size[1], tile_size, 7]
        )
        y_true = tf.transpose(y_true, perm=[0, 1, 3, 2, 4, 5])
        y_true = tf.reshape(
            y_true, [-1, window_size[0] * window_size[1], tile_size, tile_size, 7]
        )

        y_pred = tf.reshape(
            y_pred, [-1, window_size[0], tile_size, window_size[1], tile_size, 7]
        )
        y_pred = tf.transpose(y_pred, perm=[0, 1, 3, 2, 4, 5])
        y_pred = tf.reshape(
            y_pred, [-1, window_size[0] * window_size[1], tile_size, tile_size, 7]
        )

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
                for j in range(window_size[0] * window_size[1])
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

        categorical_ssim = tf.where(
            tf.math.is_nan(categorical_ssim), 1.0, categorical_ssim
        )

        return categorical_ssim

    @tf.function
    def ssim_loss_combined(self, y_true, y_pred):
  
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        mask_true = tf.math.logical_not(tf.math.logical_and(y_true >= 0, y_true <= 7))
        mask_pred = tf.math.logical_not(tf.math.logical_and(y_pred >= 0, y_pred <= 7))
        y_true = tf.where(mask_true, tf.zeros_like(y_true), y_true)
        y_pred = tf.where(mask_pred, tf.zeros_like(y_pred), y_pred)
        ssim = tf.image.ssim(  # try ssim_multiscale
            y_true,
            y_pred,
            max_val=7,
            filter_size=self.filter_size,  # try 3
        )

        ssim_loss = 1 - tf.reduce_mean(ssim)
        ssim_loss = tf.where(tf.math.is_nan(ssim_loss), 1.0, ssim_loss)

        return ssim_loss

    @tf.function
    def hybrid_loss(self, y_true, y_pred, weights=None):
        """
        Hybrid loss to minimize. Pass to model as loss during compile statement.
        It is a combination of jackard loss, focal loss and ssim loss.

        Parameters
        ----------
        :tensor y_true: ground truth mask
        :tensor y_pred: predicted mask
        :list weights: list of class weights

        Returns
        -------
        :tensor loss: hybrid loss
        """
        if weights is None:
            weights = [1 for i in range(self.n_classes)]
        jackard_loss = self.categorical_jackard_loss(y_true, y_pred)
        focal_loss = self.categorical_focal_loss(y_true, y_pred)
        ssim_loss = self.categorical_ssim_loss(y_true, y_pred)

        jf = focal_loss + jackard_loss + ssim_loss

        # tf.print(type(jd))
        # tf.print(type(ssim_loss))
        return jf * self.weights
