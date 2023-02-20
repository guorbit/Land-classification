from models.constructor import ModelGenerator, VGG16_UNET
import numpy as np
from shape import read_images
from constants import TRAINING_DATA_PATH, NUM_CLASSES
from shape_encoder import ImagePreprocessor
from models.loss_constructor import Semantic_loss_functions
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.flowreader import FlowGenerator
import os


if __name__ == "__main__":
    model = VGG16_UNET((512, 512, 3), 2)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_object = Semantic_loss_functions()
    loss_fn = loss_object.focal_loss
    # loss_fn =

    model.compile(loss_fn)
    # images, y = read_images(os.path.join(TRAINING_DATA_PATH , "x", "img")+os.sep)
    # x, masks = read_images(os.path.join(TRAINING_DATA_PATH , "y", "img")+os.sep)

   
    # convert masks to one hot encoded images
    # preprocessor = ImagePreprocessor(masks)
    # preprocessor.onehot_encode()
    # masks = preprocessor.get_encoded_images()
    # print(masks.shape)
    batch_size = 8
    generator = FlowGenerator(
        os.path.join(TRAINING_DATA_PATH, "x"),
        os.path.join(TRAINING_DATA_PATH, "y"),
        image_size=(512, 512),
        shuffle=True,
        num_classes=NUM_CLASSES,
        batch_size=8
    )
    train_generator = generator.get_generator()
    dataset_size = generator.get_dataset_size()
    print(dataset_size)

    model.fit(train_generator, epochs=1, batch_size=batch_size, steps_per_epoch=dataset_size // batch_size)
