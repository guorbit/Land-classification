"""
This file is used to debug the code. It is not used in the training process.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utilities.segmentation_utils.flowreader import (FlowGenerator,
                                                     FlowGeneratorExperimental)
from utilities.segmentation_utils.ImagePreprocessor import (
    PreprocessingQueue, random_flip_left_right, random_flip_up_down)

from constants import (NUM_CLASSES, TEST_DATA_PATH, TRAINING_DATA_PATH,
                       VALIDATION_DATA_PATH)
from models.loss_constructor import Semantic_loss_functions


def loss_debug():
    """
    Function used to debug the ms ssim  loss function.
    """
    # Define input images
    img1 = np.random.rand(16, 256, 256, 7)
    img2 = np.random.rand(16, 256, 256, 7)

    # Check input images for invalid values
    tf.debugging.check_numerics(img1, "img1 contains invalid values")
    tf.debugging.check_numerics(img2, "img2 contains invalid values")

    # Calculate MS-SSIM
    ms_ssim = tf.image.ssim_multiscale(img1, img2, max_val=1.0)


    loss_object = Semantic_loss_functions()
    loss_fn = loss_object.ssim_loss

    returned = loss_fn(img1, img2)


    # Print result
    print("ssim local:", ms_ssim)
    print("ssim loss:", returned)

def reader_debug():
    """
    Function used to debug the flowreader.
    """
    seed = 42
    image_queue = PreprocessingQueue(
        queue=[
            random_flip_up_down,
            random_flip_left_right,
            tf.image.random_brightness,
            tf.image.random_contrast,
            tf.image.random_saturation,
            # tf.image.random_hue,
        ],
        arguments=[
            {"seed": seed},
            {"seed": seed},
            {"max_delta": 0.1, "seed": seed},
            {"lower": 0.9, "upper": 1.1, "seed": seed},
            {"lower": 0.9, "upper": 1.1, "seed": seed},
            # {"max_delta": 0.2, "seed": seed},
        ],
    )
    mask_queue = PreprocessingQueue(
        queue=[
            random_flip_up_down,
            random_flip_left_right,
        ],
        arguments=[
            {"seed": seed},
            {"seed": seed},
        ],
    )
    reader_args = {
        "image_path": os.path.join(TRAINING_DATA_PATH, "x"),
        "mask_path": os.path.join(TRAINING_DATA_PATH, "y"),
        "image_size": (512, 512),  #
        "output_size": (256 * 256, 1),
        "shuffle": True,
        "preprocessing_enabled": True ,
        "channel_mask": [True,True,True],
        "num_classes": NUM_CLASSES,
        "batch_size": 1,
    }
    generator = FlowGeneratorExperimental(**reader_args)
    generator.set_preprocessing_pipeline(image_queue, mask_queue)
    generator.on_epoch_end()
    for i in range(5):
        image, mask = generator[i]

        visualize_sample(image[0,:,:, :], tf.reshape(tf.argmax(mask[0],axis=1),(256,256)))
    

def visualize_sample(image, mask):
    """
    helper function to visualize a sample
    """
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.show()




reader_debug()
