import numpy as np
import os
import keras
from models.loss_constructor import SemanticLoss
from models.callbacks import CustomReduceLROnPlateau
from keras.callbacks import TensorBoard
from map import MODEL_MAP


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
def init_loss()->SemanticLoss:   
    loss_object = SemanticLoss(weights_enabled=True)
    loss_object.set_alpha(HPARAMS["alpha"])
    loss_object.set_gamma(HPARAMS["gamma"])
    loss_object.set_window_size(HPARAMS["window_size"])
    loss_object.set_filter_size(HPARAMS["filter_size"])
    loss_object.set_filter_sigma(HPARAMS["filter_sigma"])
    loss_object.set_k1(HPARAMS["k1"])
    loss_object.set_k2(HPARAMS["k2"])
    return loss_object

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
    "loss": init_loss().categorical_focal_loss,
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

MODELS = MODEL_MAP
