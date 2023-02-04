import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np


class LossConstructor():
    weights = np.ones((7,1))
    kWeights = None
    def __init__(self,num_classes, class_counts):
        
        self.weights = self.weights.reshape((1,1,1,num_classes))
        self.kWeights = K.constant(self.weights)
    

    def weighted_cce(self,y_true, y_pred):
        yWeights = self.kWeights * y_pred         #shape (batch, 128, 128, 4)
        yWeights = K.sum(yWeights, axis=-1)  #shape (batch, 128, 128)  

        loss = K.categorical_crossentropy(y_true, y_pred) #shape (batch, 128, 128)
        wLoss = yWeights * loss

        return K.sum(wLoss, axis=(1,2))
    
    def get_function(self):
        self.weighted_cce