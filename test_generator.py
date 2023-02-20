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


def focal_loss(gamma=2.0, alpha=4.0):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.0e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1.0, model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


if __name__ == "__main__":
    model = VGG16_UNET((512, 512, 3), 2)
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
    batch_size = 8
    generator = FlowGenerator(
        os.path.join(TRAINING_DATA_PATH, "x"),
        os.path.join(TRAINING_DATA_PATH, "y"),
        batch_size=batch_size,
        image_size=(512, 512),
        shuffle=True,
        num_classes=NUM_CLASSES,
    )
    train_generator = generator.get_generator()
    dataset_size = generator.get_dataset_size()
    print(dataset_size)

    model.fit(train_generator, epochs=1, batch_size=batch_size, steps_per_epoch=dataset_size // batch_size)
