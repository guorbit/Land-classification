import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np


class LossConstructor():
    class weighted_cce(keras.losses.Loss):
        weights = np.ones((7,1))
        kWeights = None
        def __init__(self,num_classes, class_counts):
            
            self.weights = self.weights.reshape((1,1,1,num_classes))
            self.kWeights = K.constant(self.weights)
        

        def weighted_cce(self,y_true, y_pred):
            yWeights = self.kWeights * y_pred         #shape (batch, 128, 128, 4)
            yWeights = K.sum(yWeights, axis=-1)  #shape (batch, 128, 128)  

            loss = K.sparse_categorical_crossentropy(y_true, y_pred) #shape (batch, 128, 128)
            wLoss = yWeights * loss

            return K.sum(wLoss, axis=(1,2))
        
        def get_function(self):
            self.weighted_cce

    def categorical_focal_loss(alpha, gamma=2.):
        """
        Softmax version of focal loss.
        When there is a skew between different categories/labels in your data set, you can try to apply this function as a
        loss.
            m
        FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
            c=1
        where m = number of classes, c = class and o = observation
        Parameters:
        alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
        categories/labels, the size of the array needs to be consistent with the number of classes.
        gamma -- focusing parameter for modulating factor (1-p)
        Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
        References:
            Official paper: https://arxiv.org/pdf/1708.02002.pdf
            https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
        Usage:
        model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
        """

        alpha = np.array(alpha, dtype=np.float32)

        def categorical_focal_loss_fixed(y_true, y_pred):
            """
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred: A tensor resulting from a softmax
            :return: Output tensor.
            """

            # Clip the prediction value to prevent NaN's and Inf's
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

            # Calculate Cross Entropy
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate Focal Loss
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

            # Compute mean loss in mini_batch
            return K.mean(K.sum(loss, axis=-1))

        return categorical_focal_loss_fixed
