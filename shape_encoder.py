import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from constants import (
    NUM_CLASSES,
    LABEL_MAP,
)



class ImagePreprocessor():
    __images = None
    __encoded_images = None
    def __init__(self,images):
        '''
        Initializes an encoder object

        Parameters:
        ----------
        images (numpy ndarray): array of images to be encoded

        Returns:
        -------
        None
        '''
        self.__images = images

    def mount_images_new(self,images):
        '''
        Mounts a new set of images to the encoder object

        Parameters:
        ----------
        images (numpy ndarray): array of images to be encoded

        Returns:
        -------
        None
        '''

        self.__images = images

    def get_images(self):
        '''
        Returns the images that are currently stored in the encoder object

        Parameters:
        ----------
        None

        Returns:
        -------
        numpy ndarray: array of images that are currently stored in the encoder object
        '''

        return self.__images
    
    def get_encoded_images(self):
        '''
        Returns the images that are currently stored in the encoder object

        Parameters:
        ----------
        None

        Returns:
        -------
        numpy ndarray: array of images that are currently stored in the encoder object
        '''

        return self.__encoded_images

    def onehot_encode(self):
        '''
        Onehot encodes the images that are currently stored in the encoder object

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        '''


        print("Onehot encoding images...")
        self.__encoded_images = np.zeros((self.__images.shape[0], self.__images.shape[1], self.__images.shape[2], NUM_CLASSES))
        for i in tqdm(range(NUM_CLASSES)):
            self.__encoded_images[:,:,:,i] = np.where(self.__images == i, 1, 0)
        print("Onehot encoding complete.")
    



