import numpy as np
import os
from PIL import Image
from tqdm import tqdm


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
IMAGE_SIZE = (384, 384)
SUB_REGION_SIZE = (4, 4)
IMAGE_REGIONS = 0


def read_images(path):
    if (
        len(path.split("/")[0].split("_")) > 1
        and path.split("/")[0].split("_")[1] == "resized"
    ):
        print("Reading images from " + path)
        sat_images = [
            np.array(Image.open(path + f))
            for f in tqdm(os.listdir(path))
            if f.endswith(".jpg")
        ]
        print("Number of images imported: " + str(len(sat_images)))
        mask_images = [
            np.array(Image.open(path + f))
            for f in tqdm(os.listdir(path))
            if f.endswith(".png")
        ]

        # get second number from image name

        return np.array(sat_images), np.array(mask_images)
    else:

        print("Reading images from " + path)
        sat_images = [
            np.array(Image.open(path + f).resize(IMAGE_SIZE))
            for f in tqdm(os.listdir(path))
            if f.endswith(".jpg")
        ]
        print("Number of images imported: " + str(len(sat_images)))
        print("\nReading masks from " + path)
        mask_images = [
            np.array(Image.open(path + f).resize(IMAGE_SIZE))
            for f in tqdm(os.listdir(path))
            if f.endswith(".png")
        ]

        print("Number of masks imported: " + str(len(mask_images)))
        IMAGE_REGIONS = (
            len(sat_images)
            * (IMAGE_SIZE[0] * IMAGE_SIZE[1])
            / (SUB_REGION_SIZE[0] * SUB_REGION_SIZE[1])
        )
        return np.array(sat_images), np.array(mask_images)


# original datashaping
def export_images(images, masks, path):
    print("Exporting images to " + path)
    os.makedirs(os.path.join(path,"x"))
    os.makedirs(os.path.join(path,"y"))
    images = images.reshape(images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    masks = masks.reshape(images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1])

    for i in tqdm(range(images.shape[0])):

        img = Image.fromarray(images[i].astype("uint8"))
        mask = Image.fromarray(masks[i].astype("uint8"))
        img.save(os.path.join(path,"x", str(i) + ".jpg"))
        mask.save(os.path.join(path,"y", str(i) + ".png"))


def prepocess_mask_images(mask_images):
    mask_images = np.array(mask_images)
    mask_images = mask_images / 255
    mask_images = (
        mask_images[:, :, :, 0]
        + mask_images[:, :, :, 1] * 2
        + mask_images[:, :, :, 2] * 4
    )
    mask_images = np.where(mask_images > 0, mask_images - 1, mask_images)
    mask_images = mask_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    return mask_images


def preprocess_sat_images(sat_images):
    sat_images = np.array(sat_images)
    # sat_images = sat_images / 255
    sat_images = sat_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1], 3)
    return sat_images


def preprocess_train_images(images):
    images = np.array(images)
    images = images / 255

    return images

def split_read(path,val_percent):
    images,masks = read_images(path)
    mask_images = prepocess_mask_images(masks)
    export_images(images[:int(len(images)*val_percent)],mask_images[:int(len(mask_images)*val_percent)],"archive_resized/val/")
    export_images(images[int(len(images)*val_percent):],mask_images[int(len(mask_images)*val_percent):],"archive_resized/train/")

if __name__ == "__main__":
    # read images
    if os.path.isdir("archive_resized"):
        print("Resized images already exist. Importing resized images...")
        sat_images, mask_images = read_images("archive_resized/train/")
    else:
        sat_images, mask_images = split_read("archive/train/",0.2)
        # print("Resized images do not exist. Importing the original images...")
        # sat_images, mask_images = read_images("archive/train/")

        # # preprocess images
        # sat_images = preprocess_sat_images(sat_images)
        # mask_images = prepocess_mask_images(mask_images)

        # # separate labels
        # mask_images = separate_mask_labels(mask_images)
        # overlayed_images = overlap_image_masks(sat_images, mask_images)
        # print(overlayed_images[0, 0])
        # export_images(overlayed_images, "archive_resized/train/")
