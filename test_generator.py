from models.constructor import ModelGenerator, VGG16_UNET
from models.loss_constructor import LossConstructor
import numpy as np
from shape import read_images
from constants import TRAINING_DATA_PATH
from tensorflow.keras.losses import categorical_crossentropy

if __name__ == "__main__":
    model = VGG16_UNET((512,512,3), 2)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_fn = LossConstructor(7,np.array([1,1,1,1,1,1,1])).weighted_cce
    model.compile(categorical_crossentropy)
    images,y = read_images(TRAINING_DATA_PATH + "x/")
    x,masks = read_images(TRAINING_DATA_PATH + "y/")
    model.fit(images, masks, epochs=1, batch_size=2)