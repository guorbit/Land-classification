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

if __name__ == "__main__":
    with tf.device('/device:GPU:0'):

    print("Loading model " + MODEL_NAME + "_" + str(MODEL_ITERATION))
    model = load_model(
        os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5")
    )

    images, m = read_images(VALIDATION_DATA_PATH + "x/")
    i, masks = read_images(VALIDATION_DATA_PATH + "y/")

    prediction = model.predict(images)
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

            axarr[0].imshow(images[k])
            axarr[1].imshow(new_mask[k])
            axarr[2].imshow(pred_rgb)

            plt.show()
