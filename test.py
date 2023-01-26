from shape import read_images, label_maps, IMAGE_SIZE
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from constants import MODEL_NAME, MODELS, NUM_CLASSES,VALIDATION_DATA_PATH, MODEL_ITERATION
import numpy as np

if __name__ == "__main__":

    print("Loading model " + MODEL_NAME+"_"+str(MODEL_ITERATION))
    model = load_model(MODEL_NAME+"_"+str(MODEL_ITERATION) + ".h5")

    images = read_images(VALIDATION_DATA_PATH + "x/")
    masks = read_images(VALIDATION_DATA_PATH + "y/")


    prediction = model.predict(images)
    print(prediction.shape)
    prediction = prediction.reshape(
        images.shape[0],
        MODELS[MODEL_NAME]["image_size"][0],
        MODELS[MODEL_NAME]["image_size"][1],
        NUM_CLASSES,
    )
    for k, n in enumerate(prediction):
        f, axarr = plt.subplots(1, 2)
        first_prediction = np.argmax(n, axis=2)
        pred_rgb = np.zeros(
            (
                MODELS[MODEL_NAME]["image_size"][0],
                MODELS[MODEL_NAME]["image_size"][1],
                3,
            )
        )
        for i in range(MODELS[MODEL_NAME]["image_size"][0]):
            for j in range(MODELS[MODEL_NAME]["image_size"][1]):
                pred_rgb[i, j, :] = label_maps[first_prediction[i, j]]

        axarr[0].imshow(images[k])
        axarr[1].imshow(pred_rgb)

        plt.show()
