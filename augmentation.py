from tensorflow import keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def generate_common_pipeline(seed):
    """
    Function to generate the common augmentation part of the image and masks

    Parameters:
    seed (tupple of integer): the tensorflow seed parameter for the preprocessing layers

    Returns:
    keras.sequential: the common sequential augmentation model
    """

    common_transform = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"
            ),
            keras.layers.experimental.preprocessing.RandomRotation(0.2, seed=seed),
            keras.layers.experimental.preprocessing.RandomZoom(0.2, seed=seed),
        ]
    )
    return common_transform


def generate_image_pipeline(seed):
    """
    Function to generate the only image transformation layer model

    Parameters:
    seed (tupple of integer): the tensorflow seed parameter for the preprocessing layers

    Returns:
    keras.sequential: the common sequential augmentation model
    """

    image_transform = keras.Sequential(
        [keras.layers.experimental.preprocessing.RandomContrast(0.2, seed=seed)]
    )
    common_transform = generate_common_pipeline(seed)

    return keras.Sequential([common_transform, image_transform]), common_transform


# the main augmentation pipeline function
# images = numpy ndarray
# masks = numpy ndarray


def augment_images(images, masks):
    """
    Function is the main augmentation pipeline function, call this to run the pipeline

    Parameters:
    images (numpy ndarray): the satellite images
    masks (numpy ndarray): the satellite image masks

    Returns:
    numpy ndarray: the augmented images
    numpy ndarray: the augmented images' masks
    """
    tf.get_logger().setLevel('ERROR')
    image_store = []
    mask_store = []
    for i in tqdm(range(0, 3)):
       
        seed = (i, 0)
        image_pipeline, mask_pipeline = generate_image_pipeline(seed)
        augmented_images = image_pipeline(images)
        augmented_masks = mask_pipeline(masks)
        image_store.append(augmented_images)
        mask_store.append(augmented_masks)
    augmented_images = np.concatenate(image_store)
    augmented_masks = np.concatenate(mask_store)
    return augmented_images, augmented_masks
