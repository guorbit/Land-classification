from models.constructor import ModelGenerator, VGG16_UNET
from constants import TRAINING_DATA_PATH, NUM_CLASSES, VALIDATION_DATA_PATH
from models.loss_constructor import Semantic_loss_functions
import tensorflow as tf
import numpy as np
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

def masked_categorical_crossentropy(y_true, y_pred):
    '''
    Masked categorical crossentropy to ignore background pixel label 0
    Pass to model as loss during compile statement
    '''
    y_true = y_true[...,1:]
    y_pred = y_pred[...,1:]

    loss = K.categorical_crossentropy(y_true, y_pred)
    return loss 


def categorical_focal_loss(y_true,y_pred):
    
    gamma = 2.
    alpha = .25
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss


def categorical_jackard_loss(y_true,y_pred):
    '''
    Jackard loss to minimize. Pass to model as loss during compile statement
    '''

    smooth = 1e-7
    intersection = K.sum(K.abs(y_true * y_pred), axis=-2)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-2)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    return (1 - jac) * smooth


def categorical_ssim_loss(y_true,y_pred):
    '''
    SSIM loss to minimize. Pass to model as loss during compile statement
    '''
    # calculate ssim for each channel seperately
    y_true = tf.reshape(y_true,[-1,256,256,7])
    y_pred = tf.reshape(y_pred,[-1,256,256,7])


    categorical_ssim = tf.convert_to_tensor([tf.image.ssim(tf.expand_dims(y_true[...,i],-1),tf.expand_dims(y_pred[...,i],-1),max_val = 1.0,filter_size=11) for i in range(7)])
    categorical_ssim = 1 - categorical_ssim

    # swap axes 0,1
    categorical_ssim = tf.transpose(categorical_ssim, perm=[1,0])
    return categorical_ssim
    

def ssim_loss(y_true,y_pred):
    '''
    SSIM loss to minimize. Pass to model as loss during compile statement
    '''
    # calculate ssim for each channel seperately
    y_true = tf.reshape(y_true,[-1,256,256,7])
    y_pred = tf.reshape(y_pred,[-1,256,256,7])


    
    categorical_ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    # swap axes 0,1
    
    return categorical_ssim



def hybrid_loss(y_true,y_pred):
    '''
    Hybrid loss to minimize. Pass to model as loss during compile statement
    '''
    jackard_loss = categorical_jackard_loss(y_true,y_pred)
    focal_loss = categorical_focal_loss(y_true,y_pred)
    ssim_loss = categorical_ssim_loss(y_true,y_pred)

    print(jackard_loss)
    print(focal_loss)
    print(ssim_loss)


    return jackard_loss*10**7 + focal_loss/1000 + ssim_loss*10



if __name__ == "__main__":
    model = VGG16_UNET((512, 512, 3), NUM_CLASSES)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_object = Semantic_loss_functions()
    loss_fn = dice_coef_9cat_loss
    # loss_fn =


    model.compile(ssim_loss)

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
    tuning_generator = FlowGenerator(
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
    tuning_generator = tuning_generator.get_generator()
    dataset_size = generator.get_dataset_size()

    x, y = next(train_generator)
    print(x.shape)
    print(y.shape)

    print(model.output_shape())

    model.fit(
        tuning_generator,
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
