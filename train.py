""" 
This script is used to train the model. It is called from the command line.
"""
import os
import sys

import tensorflow as tf
from utilities.segmentation_utils.flowreader import (
    FlowGenerator,
    FlowGeneratorExperimental,
)
from utilities.segmentation_utils.ImagePreprocessor import (
    PreprocessingQueue,
    random_flip_left_right,
    random_flip_up_down,
)

from constants import (
    HPARAMS,
    IO_DATA,
    MODEL_FOLDER,
    MODEL_ITERATION,
    MODEL_NAME,
    MODELS,
    NUM_CLASSES,
    TEST_DATA_PATH,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from models.constructor import VGG16_UNET

#!Note: The above package is not available in the repo. It is a custom package for reading data from a folder and generating batches of data.
#!Note: it is available under guorbit/utilities on github.


def main():
    """
    The main function of the train script. Envoked through the command line.
    """
    # define forward pass
    wrapper = VGG16_UNET(
        (
            IO_DATA["input_size"][0],
            IO_DATA["input_size"][1],
            IO_DATA["bands"],
        ),
        (IO_DATA["output_size"][0], IO_DATA["output_size"][0], NUM_CLASSES),
        NUM_CLASSES,
        load_weights=False,
        dropouts=HPARAMS["dropouts"],
    )
    model = wrapper.get_model()
    model.compile(
        optimizer=HPARAMS["optimizer"],
        loss=HPARAMS["loss"],
        metrics=HPARAMS["metrics"],
    )

    # initialize image preprocessing queues
    image_queue = PreprocessingQueue(
        queue=[
            random_flip_left_right,
            random_flip_up_down,
            tf.image.random_brightness,
            tf.image.random_contrast,
            tf.image.random_saturation,
            # tf.image.random_hue,
        ],
        arguments=[
            {"seed": HPARAMS["seed"]},
            {"seed": HPARAMS["seed"]},
            {"max_delta": 0.2, "seed": HPARAMS["seed"]},
            {"lower": 0.8, "upper": 1.2, "seed": HPARAMS["seed"]},
            {"lower": 0.8, "upper": 1.2, "seed": HPARAMS["seed"]},
            # {"max_delta": 0.2, "seed": seed},
        ],
    )
    mask_queue = PreprocessingQueue(
        queue=[
            random_flip_left_right,
            random_flip_up_down,
        ],
        arguments=[
            {"seed": HPARAMS["seed"]},
            {"seed": HPARAMS["seed"]},
        ],
    )

    # dataset iterator arguments
    reader_args = {
        "image_path": os.path.join(TRAINING_DATA_PATH, "x", "img"),
        "mask_path": os.path.join(TRAINING_DATA_PATH, "y", "img"),
        "image_size": (512, 512),  #
        "output_size": (512, 512),
        "shuffle": True,
        "preprocessing_enabled": True,
        "channel_mask": [True, True, True],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
        "read_weights": False,
        "weights_path": os.path.join(TRAINING_DATA_PATH, "weights.csv"),
    }
    val_reader_args = {
        "image_path": os.path.join(VALIDATION_DATA_PATH, "x", "img"),
        "mask_path": os.path.join(VALIDATION_DATA_PATH, "y", "img"),
        "image_size": (512, 512),  #
        "output_size": (512, 512),
        "shuffle": True,
        "preprocessing_enabled": False,
        "channel_mask": [True, True, True],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
    }

    # initialize dataset iterators
    HPARAMS["dataset"] = FlowGeneratorExperimental(**reader_args)
    HPARAMS["dataset"].set_preprocessing_pipeline(image_queue, mask_queue)
    HPARAMS["validation_dataset"] = FlowGeneratorExperimental(**val_reader_args)
    HPARAMS["dataset_size"] = len(HPARAMS["dataset"])

    # set training arguments
    training_args = {
        "dataset": HPARAMS["dataset"],
        "batch_size": HPARAMS["batch_size"],
        "epochs": HPARAMS["epochs"],
        "steps_per_epoch": HPARAMS["dataset_size"],
        "learning_rate": HPARAMS["learning_rate"],
        "validation_dataset": HPARAMS["validation_dataset"],
        "validation_steps": HPARAMS["validation_steps"],
        "callbacks": HPARAMS["callbacks"],
    }

    # train model
    model.summary()
    model.train(
        **training_args,
        enable_tensorboard=True,
    )

    # save model
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    model.save(
        os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5")
    )


if __name__ == "__main__":
    # read command line flag for debug mode
    debug = False
    sysargs = sys.argv
    if len(sysargs) > 1:
        if sysargs[1] == "--debug":
            debug = True

    if debug:
        tf.config.run_functions_eagerly(True)
