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
from models.constructor import VGG16_UNET, VGG_NANO_UNET

#!Note: The above package is not available in the repo. It is a custom package for reading data from a folder and generating batches of data.
#!Note: it is available under guorbit/utilities on github.


def main(hparams, save = True, return_logs = False):
    """
    The main function of the train script. Envoked through the command line.
    """
    # define forward pass
    wrapper = VGG_NANO_UNET(
        (
            IO_DATA["input_size"][0],
            IO_DATA["input_size"][1],
            IO_DATA["bands"],
        ),
        (IO_DATA["output_size"][0], IO_DATA["output_size"][0], NUM_CLASSES),
        NUM_CLASSES,
        load_weights=False,
        dropouts=hparams["dropouts"],
    )
    model = wrapper.get_model()
    print("MODEL NAME:",model.name)
    model.compile(
        optimizer=hparams["optimizer"],
        loss=hparams["loss"],
        metrics=hparams["metrics"],
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
            {"seed": hparams["seed"]},
            {"seed": hparams["seed"]},
            {"max_delta": 0.2, "seed": hparams["seed"]},
            {"lower": 0.8, "upper": 1.2, "seed": hparams["seed"]},
            {"lower": 0.8, "upper": 1.2, "seed": hparams["seed"]},
            # {"max_delta": 0.2, "seed": seed},
        ],
    )
    mask_queue = PreprocessingQueue(
        queue=[
            random_flip_left_right,
            random_flip_up_down,
        ],
        arguments=[
            {"seed": hparams["seed"]},
            {"seed": hparams["seed"]},
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
        "batch_size": hparams["batch_size"],
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
        "batch_size": hparams["batch_size"],
    }

    # initialize dataset iterators
    hparams["dataset"] = FlowGeneratorExperimental(**reader_args)
    hparams["dataset"].set_preprocessing_pipeline(image_queue, mask_queue)
    hparams["validation_dataset"] = FlowGeneratorExperimental(**val_reader_args)
    hparams["dataset_size"] = len(hparams["dataset"])

    # set training arguments
    training_args = {
        "dataset": hparams["dataset"],
        "batch_size": hparams["batch_size"],
        "epochs": hparams["epochs"],
        "steps_per_epoch": hparams["dataset_size"],
        "learning_rate": hparams["learning_rate"],
        "validation_dataset": hparams["validation_dataset"],
        "validation_steps": hparams["validation_steps"],
        "callbacks": hparams["callbacks"],
    }

    # train model
    if not return_logs:
        model.summary()
    model.train(
        **training_args,
        enable_tensorboard=True,
    )

    if save:
    # save model
        if not os.path.isdir(MODEL_FOLDER):
            os.mkdir(MODEL_FOLDER)
        model.save(
            os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5")
        )
    if return_logs:
        return model.get_backup_logs()
    else:
        from hparam_snapshot import write_to_database, get_iteration
        logs = model.get_backup_logs()
        iteratetion = get_iteration(model.name)

        write_to_database(
            hparams,
            model.name,
            iteratetion,
            logs["val_accuracy"],
            logs["val_recall"],
            logs["val_precision"],
            logs["val_loss"],
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

    main(HPARAMS)
