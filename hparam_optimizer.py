import os

import keras
import optuna
import tensorflow as tf
from utilities.segmentation_utils.flowreader import FlowGenerator
from utilities.segmentation_utils.ImagePreprocessor import PreprocessingQueue

from constants import NUM_CLASSES, TRAINING_DATA_PATH, VALIDATION_DATA_PATH
from models.callbacks import CustomReduceLROnPlateau, accuracy_drop_callback
from models.constructor import VGG16_UNET
from models.loss_constructor import Semantic_loss_functions


def objective(trial):
    """
    The objective function for the optuna hyperparameter optimization.
    """
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

    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    seed = 42

    random_brightness = trial.suggest_float("random_brightness", 0.0, 0.5)

    random_contrast_min = trial.suggest_float("random_contrast_min", 0.5, 1)
    random_contrast_max = trial.suggest_float("random_contrast_max", 1, 1.5)

    random_saturation_min = trial.suggest_float("random_saturation_min", 0.5, 1)
    random_saturation_max = trial.suggest_float("random_saturation_max", 1, 1.5)

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
            {"max_delta": random_brightness, "seed": seed},
            {"lower": random_contrast_min, "upper": random_contrast_max, "seed": seed},
            {
                "lower": random_saturation_min,
                "upper": random_saturation_max,
                "seed": seed,
            },
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
        "epochs": 1,
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

    lr_factor = trial.suggest_categorical("lr_factor", [0.1, 0.01, 0.001])
    lr_patience = trial.suggest_int("lr_patience", 1, 5)

    reduce_lr = CustomReduceLROnPlateau(
        monitor="val_loss", factor=lr_factor, patience=lr_patience, min_lr=1e-10
    )

    generator = FlowGenerator(**reader_args)
    train_generator = generator.get_generator()
    val_generator = FlowGenerator(**val_reader_args)
    val_dataset_size = val_generator.get_dataset_size()
    dataset_size = generator.get_dataset_size()
    val_generator = val_generator.get_generator()
    

    training_args["steps_per_epoch"] = dataset_size // batch_size
    training_args["validation_steps"] = val_dataset_size // batch_size

    model.train(
        train_generator,
        **training_args,
        validation_dataset=val_generator,
        callbacks=[reduce_lr]
    )
    logs = model.get_backup_logs()
    print(logs)
    return logs["val_accuracy"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
