import os

import keras
import optuna
import tensorflow as tf


from models.loss_constructor import SemanticLoss
from keras.callbacks import TensorBoard
from train import main
from constants import reduce_lr, NUM_CLASSES, TRAINING_DATA_PATH


def objective(trial):
    loss_object = SemanticLoss(
        n_classes=NUM_CLASSES, weights_enabled=True, weights_path=TRAINING_DATA_PATH
    )

    hparams = {
        # NOTE: loss function arguments
        "gamma": 1.5,  # focal loss gamma
        "alpha": 0.25,  # focal loss alpha
        "window_size": (4, 4),  # ssim segmentation number of windows
        "filter_size": 25,  # ssim filter size
        "filter_sigma": 2.5,  # ssim filter sigma
        "k1": 0.06,  # ssim k1
        "k2": 0.02,  # ssim k2
        "weights_enabled": True,  # whether to use weights in loss function
        # NOTE: arguments for constructing the models forward pass
        "load_weights": False,  # whether to preload weights
        "dropouts": [trial.suggest_float(f"dropout {i}", 0.0, 0.4) for i in range(9)],
        # NOTE: arguments for compiling the model
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),  # optimizer
        "loss": keras.losses.CategoricalCrossentropy(),  # loss function
        "metrics": ["accuracy"],  # metrics
        # NOTE: arguments for training the model
        "batch_size": 2,  # batch size
        "seed": 42,  # random seed
        "dataset_size": None,  # dataset size
        "dataset": None,  # dataset
        "epochs": 5,  # number of epochs
        "steps_per_epoch": None,  # steps per epoch
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-6, 1e-2
        ),  # learning rate
        "validation_dataset": None,  # validation dataset
        "validation_steps": 50,  # validation steps
        "callbacks": [reduce_lr],  # callbacks
    }
    tb_callback = TensorBoard(
        log_dir=os.path.join("tb_log", "model_optimization", f"trial_{trial.number}"),
        histogram_freq=1,
        write_graph=True,
    )
    hparams["callbacks"].append(tb_callback)

    loss_object.set_alpha(hparams["alpha"])
    loss_object.set_gamma(hparams["gamma"])
    loss_object.set_window_size(hparams["window_size"])
    loss_object.set_filter_size(hparams["filter_size"])
    loss_object.set_filter_sigma(hparams["filter_sigma"])
    loss_object.set_k1(hparams["k1"])
    loss_object.set_k2(hparams["k2"])
    logs = main(hparams, save=False, return_logs=True)
    print(logs)

    return logs["val_loss"]


if __name__ == "__main__":
    storage_url = "sqlite:///example.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="model_optimization_dropout",
        storage="sqlite:///optimization.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=25, show_progress_bar=True)
    print(study.best_params)
