from tensorflow import keras
import tensorflow as tf

def generate_common_pipeline():
    common_transform = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
        keras.layers.experimental.preprocessing.RandomZoom(0.2),

    ])
    return common_transform

def generate_image_pipeline():
    image_transform = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255),
        keras.layers.experimental.preprocessing.RandomContrast(0.2)
    ])
    common_transform = generate_common_pipeline()

    return keras.Sequential([common_transform, image_transform]), common_transform 


def augment_images(images, masks):
    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)
    image_pipeline, mask_pipeline = generate_image_pipeline()
    
    
    seed = (1,0)
    augmented_images = image_pipeline(images,seed = seed)
    augmented_masks = mask_pipeline(masks, seed = seed)


    return augmented_images.numpy(), augmented_masks.numpy()