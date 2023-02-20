import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
            width_shift_range=0.1,
            height_shift_range=0.1,
        )  # custom fuction for each image you can use resnet one too.
        mask_datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
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
            target_size=self.image_size,
            color_mode = 'grayscale'
        )

        self.train_generator = zip(image_generator, mask_generator)

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
