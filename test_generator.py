from models.constructor import ModelGenerator, VGG16_UNET
from constants import TRAINING_DATA_PATH, NUM_CLASSES, TEST_DATA_PATH
from models.loss_constructor import Semantic_loss_functions
from models.callbacks import accuracy_drop_callback, CustomReduceLROnPlateau
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import (
    MODEL_NAME,
    MODELS,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    NUM_CLASSES,
    MODEL_ITERATION,
    MODEL_FOLDER,
)
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
import os
from tensorflow import keras
from utilities.segmentation_utils.flowreader import FlowGenerator
import utilities.segmentation_utils.ImagePreprocessor as ImagePreprocessor
from utilities.segmentation_utils.ImagePreprocessor import PreprocessingQueue

#!Note: The above package is not available in the repo. It is a custom package for reading data from a folder and generating batches of data.
#!Note: it is available under guorbit/utilities on github.


def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))


def dice_coef_9cat_loss(y_true, y_pred):
    """
    Dice loss to minimize. Pass to model as loss during compile statement
    """
    return 1 - dice_coef_9cat(y_true, y_pred)


def masked_categorical_crossentropy(y_true, y_pred):
    """
    Masked categorical crossentropy to ignore background pixel label 0
    Pass to model as loss during compile statement
    """
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    loss = K.categorical_crossentropy(y_true, y_pred)
    return loss


if __name__ == "__main__":
    wrapper = VGG16_UNET((512, 512, 3), (256 * 256, 7), NUM_CLASSES)
    model = wrapper.get_model()

    # model.create_model(load_weights=True)

    loss_object = Semantic_loss_functions()
    loss_fn = loss_object.hybrid_loss
    # loss_fn =

    model.compile(
        optimizer=keras.optimizers.SGD(momentum=0.8),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    batch_size = 16
    seed = 42
    image_queue = PreprocessingQueue(
        queue=[
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
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
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
        ],
        arguments=[
            {"seed": seed},
            {"seed": seed},
        ],
    )

    # print(model.output_shape())
    dataset_size = None
    training_args = {
        "batch_size": batch_size,
        "epochs": 15,
        # "steps_per_epoch": dataset_size // batch_size,
        "steps_per_epoch": dataset_size,
        # "validation_steps": 40,
        # "validation_data": tuning_generator,
    }

    reader_args = {
        "image_path": os.path.join(TRAINING_DATA_PATH, "x"),
        "mask_path": os.path.join(TRAINING_DATA_PATH, "y"),
        "image_size": (512, 512),  #
        "output_size": (256 * 256, 1),
        "shuffle": True,
        "preprocessing_enabled": True,
        "preprocessing_queue_image": image_queue,
        "preprocessing_queue_mask": mask_queue,
        "num_classes": NUM_CLASSES,
        "batch_size": batch_size,
    }

    val_reader_args = {
        "image_path": os.path.join(VALIDATION_DATA_PATH, "x"),
        "mask_path": os.path.join(VALIDATION_DATA_PATH, "y"),
        "image_size": (512, 512),  #
        "output_size": (256 * 256, 1),
        "shuffle": True,
        "preprocessing_enabled": False,
        "num_classes": NUM_CLASSES,
        "batch_size": batch_size,
    }

    reduce_lr = CustomReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=1E-10
    )

    generator = FlowGenerator(**reader_args)
    train_generator = generator.get_generator()
    val_generator = FlowGenerator(**val_reader_args)
    dataset_size = generator.get_dataset_size()
    val_generator = val_generator.get_generator()
    training_args["steps_per_epoch"] = dataset_size // batch_size

    model.train(
        train_generator,
        **training_args,
        validation_dataset=val_generator,
        validation_steps=30,
        callbacks=[reduce_lr]
    )

    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    model.save(
        os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5")
    )
