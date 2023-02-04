import tensorflow as tf
from tensorflow import keras



class ModelGenerator():
    name = None
    encoder = None
    decoder = None
    input_shape = None
    output_shape = None
    n_classes = None

    def __init__(self,encoder,decoder, input_shape, n_classes):
        self.name = encoder+"_"+decoder
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.n_classes = n_classes

    def summary(self):
        pass

    def train():
        pass

    def predict():
        pass

    def evaluate():
        pass
        
