from models.constructor import ModelGenerator, VGG16_UNET
from constants import TRAINING_DATA_PATH, NUM_CLASSES, VALIDATION_DATA_PATH
from models.loss_constructor import Semantic_loss_functions
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import (
    MODEL_NAME,
    MODELS,
    TRAINING_DATA_PATH,
    NUM_CLASSES,
    MODEL_ITERATION,
    MODEL_FOLDER,
)


from keras import backend as K
import os
from tensorflow import keras
from utilities.segmentation_utils.flowreader import FlowGenerator
import utilities.segmentation_utils.ImagePreprocessor as ImagePreprocessor
#!Note: The above package is not available in the repo. It is a custom package for reading data from a folder and generating batches of data.
#!Note: it is available under guorbit/utilities on github.

def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_9cat(y_true, y_pred)


if __name__ == "__main__":
    model = VGG16_UNET((512, 512, 3), NUM_CLASSES)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_object = Semantic_loss_functions()
    loss_fn = dice_coef_9cat_loss
    # loss_fn =


    model.compile(loss_fn)

    batch_size = 4
    generator = FlowGenerator(
        os.path.join(TRAINING_DATA_PATH, "x"),
        os.path.join(TRAINING_DATA_PATH, "y"),
        image_size=(512, 512),#
        output_size=(256*256,1),
        shuffle=True,
        preprocessing_enabled=True,
        num_classes=NUM_CLASSES,
        batch_size=batch_size,
    )

    train_generator = generator.get_generator()
    dataset_size = generator.get_dataset_size()

    x, y = next(train_generator)
    print(x.shape)
    print(y.shape)

    print(model.output_shape())

    model.fit(
        train_generator,
        epochs=20,
        batch_size=batch_size,
        steps_per_epoch=dataset_size // batch_size,
    )
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    model.save(
        os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5")
    )
