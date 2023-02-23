from models.constructor import ModelGenerator, VGG16_UNET
import numpy as np
from shape import read_images
from constants import TRAINING_DATA_PATH, NUM_CLASSES,VALIDATION_DATA_PATH
from shape_encoder import ImagePreprocessor
from models.loss_constructor import Semantic_loss_functions
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import MODEL_NAME, MODELS, TRAINING_DATA_PATH, NUM_CLASSES, MODEL_ITERATION, MODEL_FOLDER
from models.flowreader import FlowGenerator
from keras import backend as K
import os
from tensorflow import keras


def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred):
    dice = [0, 0, 0, 0, 0]
    for index in range(NUM_CLASSES):
        dice -= dice_coef(y_true[:, index, :], y_pred[:, index, :])
    return dice


gamma = 2.0
alpha = 0.25


def focal_loss(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss


def categorical_focal_loss(y_true, y_pred):

    focal = [0, 0, 0, 0, 0, 0, 0]
    for index in range(NUM_CLASSES):
        focal -= focal_loss(y_true[:, index, :], y_pred[:, index, :])

    return focal

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


if __name__ == "__main__":
    model = VGG16_UNET((512, 512, 3), NUM_CLASSES)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_object = Semantic_loss_functions()
    loss_fn = loss_object.dice_loss
    # loss_fn =

    model.compile(loss_fn)
    # images, y = read_images(os.path.join(TRAINING_DATA_PATH , "x", "img")+os.sep)
    # x, masks = read_images(os.path.join(TRAINING_DATA_PATH , "y", "img")+os.sep)

    # convert masks to one hot encoded images
    # preprocessor = ImagePreprocessor(masks)
    # preprocessor.onehot_encode()
    # masks = preprocessor.get_encoded_images()
    # print(masks.shape)
    batch_size = 4
    generator = FlowGenerator(
        os.path.join(TRAINING_DATA_PATH, "x"),
        os.path.join(TRAINING_DATA_PATH, "y"),
        image_size=(512, 512),
        shuffle=True,
        num_classes=NUM_CLASSES,
        batch_size=batch_size,
    )

    train_generator = generator.get_generator()
    dataset_size = generator.get_dataset_size()




    x, y = next(train_generator)
    print(x.shape)
    print(y.shape)


    print(dataset_size)

    model.fit(
        train_generator,
        epochs=5,
        batch_size=batch_size,
        steps_per_epoch=dataset_size // batch_size,
   
    )
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    model.save(os.path.join(MODEL_FOLDER,MODEL_NAME+"_"+str(MODEL_ITERATION) + ".h5"))
