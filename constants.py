import numpy as np
import os
from map import MODEL_MAP


ARCHIVE_DATA_PATH = os.path.join("archive", "train")

MODEL_ITERATION = 5
MODEL_NAME = "unknown"
MODEL_FOLDER = "exported_models"
MODEL_LIBRARY = "models"

TRAINING_DATA_PATH = os.path.join("archive_resized", "train")
VALIDATION_DATA_PATH = os.path.join("archive_resized", "val")

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

MODELS = MODEL_MAP