from shape import read_images, IMAGE_SIZE
from keras_segmentation.models.unet import vgg_unet, resnet50_unet, mobilenet_unet
from keras_segmentation.models.segnet import (
    vgg_segnet,
    resnet50_segnet,
    mobilenet_segnet,
)
from keras_segmentation.models.fcn import fcn_32_vgg, fcn_32_resnet50, fcn_32_mobilenet
from constants import MODEL_NAME, MODELS, TRAINING_DATA_PATH, NUM_CLASSES, MODEL_ITERATION
import os


def create_model():
    model = MODELS[MODEL_NAME]["model"](
        n_classes=NUM_CLASSES,
        input_height=MODELS[MODEL_NAME]["image_size"][0],
        input_width=MODELS[MODEL_NAME]["image_size"][1],
    )
    return model


def train_model(model, images, masks):
    model.train(
        train_images=images,
        train_annotations=masks,
        checkpoints_path=os.path.join("checkpoints", MODEL_NAME+"_"+str(MODEL_ITERATION)),
        epochs=20,
    )
    return model


if __name__ == "__main__":

    print("Training model " + MODEL_NAME+"_"+str(MODEL_ITERATION))
    image_path = TRAINING_DATA_PATH + "x/"
    mask_path = TRAINING_DATA_PATH + "y/"

    model = create_model()
    model = train_model(model, image_path, mask_path)
    model.save(MODEL_NAME+"_"+str(MODEL_ITERATION) + ".h5")
