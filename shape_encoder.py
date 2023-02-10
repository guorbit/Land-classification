import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from constants import (
    NUM_CLASSES,
    LABEL_MAP,
)



class ImagePreprocessor():
    
    @classmethod
    def onehot_encode(self,masks):
        '''
        Onehot encodes the images that are currently stored in the encoder object

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        '''
        new_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], NUM_CLASSES))
        for i in range(NUM_CLASSES):
            new_masks[:,:,:,i] = np.where(masks == i, 1, 0)
      
    



