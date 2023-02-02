from shape import read_images, IMAGE_SIZE
from keras_segmentation.models.unet import vgg_unet, resnet50_unet, mobilenet_unet
from keras_segmentation.models.segnet import (
    vgg_segnet,
    resnet50_segnet,
    mobilenet_segnet,
)
from keras_segmentation.models.fcn import fcn_32_vgg, fcn_32_resnet50, fcn_32_mobilenet
from constants import (
    MODEL_NAME,
    MODELS,
    TRAINING_DATA_PATH,
    NUM_CLASSES,
    MODEL_ITERATION,
    MODEL_FOLDER,
)
import os
import tensorflow as tf


def create_model():
    """
    This function generates the chosen model from the given configuration

    Parameters:
    no parameters

    Returns:
    keras model
    """

    model = MODELS[MODEL_NAME]["model"](
        n_classes=NUM_CLASSES,
        input_height=MODELS[MODEL_NAME]["image_size"][0],
        input_width=MODELS[MODEL_NAME]["image_size"][1],
    )
    return model


def train_model(model, images, masks):
    """
    This function runs the main training of the model

    Parameters:
    model (keras model): the model to train
    images (string): path to training images
    masks (string): path to training mask images

    Returns:
    keras model: the trained model

    """
    if not os.path.isdir(
        os.path.join("checkpoints", MODEL_NAME + "_" + str(MODEL_ITERATION))
    ):
        os.makedirs(
            os.path.join("checkpoints", MODEL_NAME + "_" + str(MODEL_ITERATION))
        )
    model.train(
        train_images=images,
        train_annotations=masks,
        checkpoints_path=os.path.join(
            "checkpoints",
            MODEL_NAME + "_" + str(MODEL_ITERATION),
            MODEL_NAME + "_" + str(MODEL_ITERATION),
        ),
        epochs=20,
    )
    return model


if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        print("Training model " + MODEL_NAME+"_"+str(MODEL_ITERATION))
        image_path = TRAINING_DATA_PATH + "x/"
        mask_path = TRAINING_DATA_PATH + "y/"

        model = create_model()
        model = train_model(model, image_path, mask_path)
        if not os.path.isdir(MODEL_FOLDER):
            os.mkdir(MODEL_FOLDER)
        model.save(os.path.join(MODEL_FOLDER,MODEL_NAME+"_"+str(MODEL_ITERATION) + ".h5"))
