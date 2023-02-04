import tensorflow as tf
from tensorflow import keras


class ModelGenerator():
    name = None
    input_shape = None
    output_shape = None
    n_classes = None

    def __init__(self,name, input_shape, output_shape, n_classes):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_classes = n_classes
        
