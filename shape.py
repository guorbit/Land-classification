import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from constants import (
    NUM_CLASSES,
    LABEL_MAP,
    MODELS,
    MODEL_NAME,
    ARCHIVE_DATA_PATH,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
)
import tensorflow as tf


# read all images from archive/train/ using PIL
# (this is only for demonstration, you should use tf.data)
label_maps = {
    5: np.array([0, 255, 255]),
    2: np.array([255, 255, 0]),
    4: np.array([255, 0, 255]),
    1: np.array([0, 255, 0]),
    3: np.array([0, 0, 255]),
    6: np.array([255, 255, 255]),
    0: np.array([0, 0, 0]),
}
NUM_CLASSES = len(label_maps.keys())
IMAGE_SIZE = (0, 0)

def extract_num(image_name):
    # extract number from image name
        if os.path.isdir("archive_resized"):
            return int(image_name.split(".")[0])
        else:
            return int(image_name.split("_")[0])
       

def read_images(path):
    if (len(path.split("/")[0].split("_")) > 1 and path.split("/")[0].split("_")[1] == "resized"):
        path_image=path+"x/"
        path_mask=path+"y/"

        if READ_LIMIT:
            image_list=sorted(os.listdir(path_image)[:int(READ_LIMIT)],key=extract_num)
            mask_list=sorted(os.listdir(path_mask)[:int(READ_LIMIT)],key=extract_num)
        else:
            image_list=sorted(os.listdir(path_image),key=extract_num)
            mask_list=sorted(os.listdir(path_mask),key=extract_num)
        
        #Read images
        print("Reading images from " + path_image)
        sat_images = [np.array(Image.open(path_image+f)) for f in tqdm(image_list) if f.endswith(".jpg")]
        print("Number of images imported: " + str(len(sat_images)))

        #Read masks
        print("Reading images from " + path_mask)
        mask_images = [ np.array(Image.open(path_mask + f))for f in tqdm(mask_list)if f.endswith(".png")]
        print("Number of images imported: " + str(len(mask_images)))

    else:
        if READ_LIMIT:
            image_list=sorted(os.listdir(path)[:int(READ_LIMIT)],key=extract_num)
        else:
            image_list=sorted(os.listdir(path),key=extract_num)
        
        #Read images
        print("Reading images from " + path)
        sat_images = [np.array(Image.open(path + f).resize(IMAGE_SIZE)) for f in tqdm(image_list) if f.endswith(".jpg")]
        print("Number of images imported: " + str(len(sat_images)))
        
        #Read masks
        print("\nReading masks from " + path)
        mask_images = [np.array(Image.open(path + f).resize(IMAGE_SIZE))for f in tqdm(image_list) if f.endswith(".png")]
        print("Number of masks imported: " + str(len(mask_images)))

    return np.array(sat_images), np.array(mask_images)


# original datashaping
def export_images(images, masks, path):
    print("Exporting images to " + path)
    os.makedirs(os.path.join(path, "x"))
    os.makedirs(os.path.join(path, "y"))
    # images = images.reshape(images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    # masks = masks.reshape(images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1])

    for i in tqdm(range(images.shape[0])):
        img = Image.fromarray(images[i].astype("uint8"))
        mask = Image.fromarray(masks[i].astype("uint8"))

        img.save(os.path.join(path, "x", str(i) + ".jpg"))
        mask.save(os.path.join(path, "y", str(i) + ".png"))


def prepocess_mask_images(mask_images):
    mask_images = np.array(mask_images)
    mask_images = mask_images / 255
    
    mask_images = (
        mask_images[:, :, :, 0]
        + mask_images[:, :, :, 1] * 2
        + mask_images[:, :, :, 2] * 4
    )
    mask_images = np.where(mask_images > 0, mask_images - 1, mask_images)
    #mask_images = mask_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    return mask_images


def preprocess_sat_images(sat_images):
    sat_images = np.array(sat_images)
    sat_images = sat_images / 255
    #sat_images = sat_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1], 3)
    return sat_images


def preprocess_train_images(images):
    images = np.array(images)
    images = images / 255
    return images


def split_read(path, val_percent):
    images, masks = read_images(path)
    mask_images = prepocess_mask_images(masks)
    export_images(
        images[: int(len(images) * val_percent)],
        mask_images[: int(len(mask_images) * val_percent)],
        VALIDATION_DATA_PATH,
    )
    export_images(
        images[int(len(images) * val_percent) :],
        mask_images[int(len(mask_images) * val_percent) :],
        TRAINING_DATA_PATH,
    )


if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        READ_LIMIT =input("How many images do you want to read?(Leave blank for all)")
        IMAGE_SIZE = MODELS[MODEL_NAME]["image_size"]
        
        # read images
        if os.path.isdir("archive_resized"):
            print("Resized images already exist. Importing resized images...")
            sat_images, mask_images = read_images(TRAINING_DATA_PATH)
        else:
            split_read(ARCHIVE_DATA_PATH, 0.2)