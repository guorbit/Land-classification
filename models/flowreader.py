import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator,Iterator
from shape_encoder import ImagePreprocessor
import numpy as np


class FlowGenerator:
    def __init__(
        self,
        image_path,
        mask_path,
        image_size,
        num_classes,
        shuffle=True,
        batch_size = 32,
  
    ):
        """
        Initializes the flow generator object

        Parameters:
        ----------
        image (string): path to the image directory
        mask (string): path to the mask directory
        batch_size (int): batch size
        image_size (tuple): image size
        num_classes (int): number of classes
        shuffle (bool): whether to shuffle the dataset or not

        Returns:
        -------
        None

        """

        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.__make_generator()
        print("Reading images from: ", self.image_path)

    def get_dataset_size(self):
        """
        Returns the length of the dataset

        Parameters:
        ----------
        None

        Returns:
        -------
        int: length of the dataset

        """

        return len(os.listdir(os.path.join(self.image_path,"img")))

    def __make_generator(self):
        """
        Creates the generator

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        """
        seed = 909  # (IMPORTANT) to transform image and corresponding mask with same augmentation parameter.
        image_datagen = ImageDataGenerator(
         
        )  # custom fuction for each image you can use resnet one too.
        mask_datagen = ImageDataGenerator(
            
        )  # to make mask as feedable formate (256,256,1)

        image_generator = image_datagen.flow_from_directory(
            self.image_path,
            class_mode=None,
            seed=seed,
            batch_size=self.batch_size,
            target_size=self.image_size,
        )

        mask_generator = mask_datagen.flow_from_directory(
            self.mask_path,
            class_mode=None,
            seed=seed,
            batch_size=self.batch_size,
            target_size=(self.image_size[0]//2,self.image_size[1]//2),
            color_mode = 'grayscale'
        )
        mask_generator = self.preprocess_mask(mask_generator)

        self.train_generator = zip(image_generator, mask_generator)
        self.train_generator = self.preprocess(self.train_generator)

    def get_generator(self):
        """
        Returns the generator

        Parameters:
        ----------
        None

        Returns:
        -------
        generator: generator object

        """

        return self.train_generator
    
    def preprocess_mask(self,generator):
        for batch in generator:
            yield batch

    
    def preprocess(self,generator_zip):
        for (img,mask) in generator_zip:
            for i in range(len(img)):
                seed = np.random.randint(0, 1000)
                preprocessor,image_preprocessor = self.init_pipeline(seed)
                img[i] = self.augmentation_pipeline(img[i],image_preprocessor)
                mask[i] = self.augmentation_pipeline(mask[i],preprocessor)
            yield (img,mask)



    def init_pipeline(self,seed):
        preprocessor = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",seed=seed),
            keras.layers.experimental.preprocessing.RandomRotation(0.2,seed=seed),
            keras.layers.experimental.preprocessing.RandomZoom(0.2,seed=seed),
            #keras.layers.experimental.preprocessing.RandomCrop(256,256,seed=seed),
            keras.layers.experimental.preprocessing.RandomTranslation(0.2,0.2,seed=seed),
        ])
        image_preprocessor = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomContrast(0.2),
            #keras.layers.experimental.preprocessing.RandomBrightness(0.2),
            #keras.layers.experimental.preprocessing.RandomSaturation(0.2),
            #keras.layers.experimental.preprocessing.RandomHue(0.2),
        ])
        image_preprocessor.add(preprocessor)
        return preprocessor,image_preprocessor



    def augmentation_pipeline(self,images,pipeline):
        images = pipeline(images)
        return images
 