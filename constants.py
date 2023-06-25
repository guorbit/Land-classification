"""
Here we define all the constants that are used in the project, 
including:
- hyperparameters
- model names
- paths to data
- callbacks
- loss function
"""
import os

import keras
import numpy as np
from keras.callbacks import TensorBoard

from map import MODEL_MAP
from models.callbacks import CustomReduceLROnPlateau
from models.loss_constructor import SemanticLoss

ARCHIVE_TRAIN_DATA_PATH = os.path.join("archive", "train")
ARCHIVE_VAL_DATA_PATH = os.path.join("archive", "val")
ARCHIVE_TEST_DATA_PATH = os.path.join("archive", "test")

MODEL_ITERATION = 6
MODEL_NAME = "unknown"
MODEL_FOLDER = "exported_models"
MODEL_LIBRARY = "models"

TRAINING_DATA_PATH = os.path.join("archive_resized", "train")
VALIDATION_DATA_PATH = os.path.join("archive_resized", "val")
TEST_DATA_PATH = os.path.join("archive_resized", "test")

NUM_CLASSES = 7

LABEL_MAP = {
    5: np.array([0, 255, 255]),
    2: np.array([255, 255, 0]),
    4: np.array([255, 0, 255]),
    1: np.array([0, 255, 0]),
    3: np.array([0, 0, 255]),
    6: np.array([255, 255, 255]),
    0: np.array([0, 0, 0]),
}

reduce_lr = CustomReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=1e-10
)
tb_callback = TensorBoard(
    log_dir=os.path.join("tb_log", MODEL_NAME + "_" + str(MODEL_ITERATION)),
    histogram_freq=1,
    write_graph=True,
)

loss_object = SemanticLoss(n_classes=NUM_CLASSES,weights_enabled=True, weights_path=TRAINING_DATA_PATH)

HPARAMS = {
    # NOTE: loss function arguments
    "gamma": 1.5, # focal loss gamma
    "alpha": 0.25, # focal loss alpha
    "window_size": (4, 4), # ssim segmentation number of windows
    "filter_size": 25, # ssim filter size
    "filter_sigma": 2.5, # ssim filter sigma
    "k1": 0.06, # ssim k1
    "k2": 0.02, # ssim k2
    "weights_enabled": True, # whether to use weights in loss function
    # NOTE: arguments for constructing the models forward pass
    "load_weights": False, # whether to preload weights
    "dropouts": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # NOTE: arguments for compiling the model
    "optimizer": keras.optimizers.Adam(learning_rate=0.001), # optimizer
    "loss": loss_object.categorical_focal_loss,  # loss function
    "metrics": ["accuracy"], # metrics
    # NOTE: arguments for training the model
    "batch_size": 4, # batch size
    "seed": 42, # random seed
    "dataset_size": None, # dataset size
    "dataset": None, # dataset
    "epochs": 5, # number of epochs
    "steps_per_epoch": None, # steps per epoch
    "learning_rate": 1e-5, # learning rate
    "validation_dataset": None, # validation dataset
    "validation_steps": 50, # validation steps
    "callbacks": [reduce_lr, tb_callback], # callbacks
}

loss_object.set_alpha(HPARAMS["alpha"])
loss_object.set_gamma(HPARAMS["gamma"])
loss_object.set_window_size(HPARAMS["window_size"])
loss_object.set_filter_size(HPARAMS["filter_size"])
loss_object.set_filter_sigma(HPARAMS["filter_sigma"])
loss_object.set_k1(HPARAMS["k1"])
loss_object.set_k2(HPARAMS["k2"])

MODELS = MODEL_MAP

IO_DATA = {
    "input_size": MODELS[MODEL_NAME]["image_size"],
    "bands": 3,
    "output_size": MODELS[MODEL_NAME]["output_size"],
    "num_classes": NUM_CLASSES,
}


