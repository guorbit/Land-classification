import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from constants import (
    NUM_CLASSES,
    LABEL_MAP,
)
import tensorflow as tf



class ImagePreprocessor():
    
    @classmethod
    def onehot_encode(self,masks):
        '''
        Onehot encodes the images coming from the image generator object

        Parameters:
        ----------
        masks (tf tensor): masks to be onehot encoded

        Returns:
        -------
        None
        '''
        if masks.shape[-1] != None:
            return tf.one_hot(tf.squeeze(masks, NUM_CLASSES))
        else:

            return masks 
    

    @classmethod
    def get_encoded_images(self):
        '''
        Returns the onehot encoded images

        Parameters:
        ----------
        None

        Returns:
        -------
        numpy array
        '''
        return self.encoded_images
      
    



