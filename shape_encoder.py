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
    def onehot_encode(self,masks, image_size, num_classes):
        '''
        Onehot encodes the images coming from the image generator object

        Parameters:
        ----------
        masks (tf tensor): masks to be onehot encoded

        Returns:
        -------
        None
        '''
        encoded = np.zeros((masks.shape[0],image_size[0]//2*image_size[1]//2,num_classes))
        for i in range(num_classes):
            encoded[:,:,i] = tf.squeeze((masks == i).astype(int))
    
        return encoded

      
    



