from shape import read_images, IMAGE_SIZE
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt
from constants import (
    MODEL_NAME,
    MODELS,
    NUM_CLASSES,
    VALIDATION_DATA_PATH,
    MODEL_ITERATION,
    LABEL_MAP,
    MODEL_FOLDER,
)
import numpy as np
import os
import tensorflow as tf
from models.loss_constructor import Semantic_loss_functions
from test_generator import dice_coef_9cat_loss
from test_generator import (
    masked_categorical_crossentropy,
    categorical_focal_loss,
    categorical_jackard_loss,
    hybrid_loss,
    categorical_ssim_loss,
)
from PIL import Image

if __name__ == "__main__":
    with tf.device("/device:GPU:0"):
        loss = Semantic_loss_functions()
        unet3p_hybrid_loss = loss.unet3p_hybrid_loss
        print("Loading model " + MODEL_NAME + "_" + str(MODEL_ITERATION))
        model = load_model(
            os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5"),
            custom_objects={
                "dice_coef_9cat_loss": dice_coef_9cat_loss,
                "masked_categorical_crossentropy": masked_categorical_crossentropy,
                "categorical_focal_loss": categorical_focal_loss,
                "categorical_jackard_loss": categorical_jackard_loss,
                "hybrid_loss": hybrid_loss,
                "categorical_ssim_loss": categorical_ssim_loss,
            },
        )

        n = len(os.listdir(os.path.join(VALIDATION_DATA_PATH, "x", "img")))
        images, m = read_images(os.path.join(VALIDATION_DATA_PATH, "x", "img"))
        i, masks = read_images(os.path.join(VALIDATION_DATA_PATH, "y", "img"))
        print("SHAPE  - - - - - - - ", images.shape)
        prediction = model.predict(images)
        print(prediction.shape)
        new_mask = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3))
        print(prediction.shape)
        prediction = prediction.reshape(
            images.shape[0],
            MODELS[MODEL_NAME]["output_size"][0],
            MODELS[MODEL_NAME]["output_size"][1],
            NUM_CLASSES,
        )
        for k, n in enumerate(prediction):
            f, axarr = plt.subplots(1, 3)
            first_prediction = np.argmax(n, axis=2)
            pred_rgb = np.zeros(
                (
                    MODELS[MODEL_NAME]["output_size"][0],
                    MODELS[MODEL_NAME]["output_size"][1],
                    3,
                )
            )
            for i in range(MODELS[MODEL_NAME]["output_size"][0]):
                for j in range(MODELS[MODEL_NAME]["output_size"][1]):
                    pred_rgb[i, j, :] = LABEL_MAP[first_prediction[i, j]]
            for i in range(masks.shape[1]):
                for j in range(masks.shape[2]):
                    new_mask[k, i, j, :] = LABEL_MAP[masks[k, i, j]]

            export_mode = False
            axarr[0].imshow(images[k])
            axarr[1].imshow(new_mask[k])
            axarr[2].imshow(pred_rgb)
            plt.show()
            if export_mode:
                inp = input("Enter to continue, y to save")
                if inp == "y":
                    plt.imshow(pred_rgb / 255)
                    plt.axis("off")
                    plt.savefig("test.png", bbox_inches="tight")
                    plt.imshow(new_mask[k] / 255)
                    plt.savefig("test2.png", bbox_inches="tight")
                    plt.imshow(images[k] / 255)
                    plt.savefig("test3.png", bbox_inches="tight")
