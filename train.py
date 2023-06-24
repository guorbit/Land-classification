import sys
from models.constructor import ModelGenerator, VGG16_UNET
from constants import TRAINING_DATA_PATH, NUM_CLASSES, TEST_DATA_PATH
from models.loss_constructor import SemanticLoss
from models.callbacks import accuracy_drop_callback, CustomReduceLROnPlateau
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
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
from keras.callbacks import TensorBoard
import os
from tensorflow import keras
from utilities.segmentation_utils.flowreader import (
    FlowGenerator,
    FlowGeneratorExperimental,
)
import utilities.segmentation_utils.ImagePreprocessor as ImagePreprocessor
from utilities.segmentation_utils.ImagePreprocessor import PreprocessingQueue

#!Note: The above package is not available in the repo. It is a custom package for reading data from a folder and generating batches of data.
#!Note: it is available under guorbit/utilities on github.


if __name__ == "__main__":
    # read command line flag for debug mode
    debug = False
    sysargs = sys.argv
    if len(sysargs) > 1:
        if sysargs[1] == "--debug":
            debug = True

    if debug:
        tf.config.run_functions_eagerly(True)

    IO_IMAGE_SIZE = (512, 512)
    BANDS = 3

    reduce_lr = CustomReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=1e-10
    )
    # save_mask = SavePredictedMaskCallback("logs\images")
    tb_callback = TensorBoard(
        log_dir=os.path.join("tb_log", MODEL_NAME + "_" + str(MODEL_ITERATION)),
        histogram_freq=1,
        write_graph=True,
    )

    # initialize loss function
    loss_object = SemanticLoss(weights_enabled=True)
    DEFAULT_DATA = {
        "input_size": (512, 512),
        "bands": 3,
        "output_size": (512, 512),
        "num_classes": NUM_CLASSES,
    }

    HPARAMS = {
        # NOTE: loss function arguments
        "gamma": 1.4,
        "alpha": 0.25,
        "window_size": (4, 4),
        "filter_size": 25,
        "filter_sigma": 2.5,
        "k1": 0.06,
        "k2": 0.02,
        "weights_enabled": True,
        # NOTE: arguments for constructing the models forward pass
        "load_weights": False,
        "dropouts": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # NOTE: arguments for compiling the model
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "loss": loss_object.categorical_focal_loss,
        "metrics": ["accuracy"],
        # NOTE: arguments for training the model
        "batch_size": 4,
        "seed": 42,
        "dataset_size": None,
        "dataset": None,
        "epochs": 5,
        "steps_per_epoch": None,
        "learning_rate": 1e-5,
        "validation_dataset": None,
        "validation_steps": 50,
        "callbacks": [reduce_lr, tb_callback],
    }
    loss_object.set_alpha(HPARAMS["alpha"])
    loss_object.set_gamma(HPARAMS["gamma"])
    loss_object.set_window_size(HPARAMS["window_size"])
    loss_object.set_filter_size(HPARAMS["filter_size"])
    loss_object.set_filter_sigma(HPARAMS["filter_sigma"])
    loss_object.set_k1(HPARAMS["k1"])
    loss_object.set_k2(HPARAMS["k2"])

    wrapper = VGG16_UNET(
        (
            DEFAULT_DATA["input_size"][0],
            DEFAULT_DATA["input_size"][1],
            DEFAULT_DATA["bands"],
        ),
        (DEFAULT_DATA["output_size"][0], DEFAULT_DATA["output_size"][0], NUM_CLASSES),
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
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
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
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
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
