import numpy as np
import os
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm
from constants import (
    NUM_CLASSES,
    LABEL_MAP,
    MODELS,
    MODEL_NAME,
    ARCHIVE_TEST_DATA_PATH,
    ARCHIVE_TRAIN_DATA_PATH,
    ARCHIVE_VAL_DATA_PATH,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
)
import tensorflow as tf
from utilities.transform_utils.image_cutting import cut_ims_in_directory
from utilities.transform_utils.image_stats import get_distribution_seg

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


def read_images(path, READ_LIMIT=None):
    test_x = os.path.join(TEST_DATA_PATH, "x", "img")
    test_y = os.path.join(TEST_DATA_PATH, "y", "img")

    IMAGE_SIZE = MODELS[MODEL_NAME]["image_size"]
    if READ_LIMIT:
        image_list = sorted(os.listdir(path)[: int(READ_LIMIT)])
    else:
        image_list = sorted(os.listdir(path))

    # Read images
    print("Reading images from " + path)
    sat_images = [
        np.array(Image.open(os.path.join(test_x, f)).resize(IMAGE_SIZE))
        for f in tqdm(image_list)
    ]
    print("Number of images imported: " + str(len(sat_images)))

    # Read masks
    print("\nReading masks from " + path)
    mask_images = [
        np.array(Image.open(os.path.join(test_y, f)).resize(IMAGE_SIZE))
        for f in tqdm(image_list)
    ]
    print("Number of masks imported: " + str(len(mask_images)))

    return np.array(sat_images), np.array(mask_images)


# original datashaping
def export_images(images, masks, path):
    print("Exporting images to " + path)
    os.makedirs(os.path.join(path, "x", "img"))
    os.makedirs(os.path.join(path, "y", "img"))

    for i in tqdm(range(images.shape[0])):
        img = Image.fromarray(images[i].astype("uint8"))
        mask = Image.fromarray(masks[i].astype("uint8"))
        img.save(os.path.join(path, "x", "img", str(i) + ".jpg"))
        mask.save(os.path.join(path, "y", "img", str(i) + ".png"))


def prepocess_mask_images(mask_images):
    mask_images = np.array(mask_images)
    mask_images = mask_images / 255

    mask_images = (
        mask_images[:, :, :, 0]
        + mask_images[:, :, :, 1] * 2
        + mask_images[:, :, :, 2] * 4
    )
    mask_images = np.where(mask_images > 0, mask_images - 1, mask_images)
    # mask_images = mask_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    return mask_images


def preprocess_sat_images(sat_images):
    sat_images = np.array(sat_images)
    sat_images = sat_images / 255
    # sat_images = sat_images.reshape(803, IMAGE_SIZE[0] * IMAGE_SIZE[1], 3)
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
        TEST_DATA_PATH,
    )
    export_images(
        images[int(len(images) * val_percent) :],
        mask_images[int(len(mask_images) * val_percent) :],
        TRAINING_DATA_PATH,
    )

def class_tfidf_vectorizer(path):
    df = pd.read_csv(
            os.path.join(path, "distribution.csv"), header=None
    ).iloc[:,2]
    files = os.listdir(os.path.join(path, "y", "img"))
    n = len(files)
    weights = np.ones((n,NUM_CLASSES), dtype=np.float32)
    # invers_doc_freqs = np.zeros((NUM_CLASSES,), dtype=np.float32)
    # for i in range(NUM_CLASSES):
    #     invers_doc_freqs[i] = math.log(n/df[i],100)

    # mean = np.mean(invers_doc_freqs)
    # diffs = (invers_doc_freqs - mean)/2
    # invers_doc_freqs = invers_doc_freqs - diffs
    # invers_doc_freqs = invers_doc_freqs/math.sqrt(np.sum(invers_doc_freqs**2))
    # print("invers doc freqs:",invers_doc_freqs)
    for i,file in enumerate(tqdm(files)):
        mask = np.array(Image.open(os.path.join(path,"y","img", file)))
        length = 0
        # calculate mean of both tf and idf separately
        tfs = np.zeros((NUM_CLASSES,), dtype=np.float32)
        idfs = np.zeros((NUM_CLASSES,), dtype=np.float32)

        for j in range(NUM_CLASSES):
            tfs[j] = (np.sum(mask == j))/(mask.shape[0]*mask.shape[1])
            # term_frequency = -math.log2(term_frequency) if term_frequency > 0 else 0
            idfs[j] = math.log(n/df[j],10) 
            # print(f"class {j},term frequency: {term_frequency}, inverse doc freq: {invers_doc_freq}")
            length += weights[i,j]**2
        mean = np.mean(idfs[tfs>0])
        # print(mean)
        diff = np.where(mean>0,idfs-mean,0)*99/100
        weights[i,:] = tfs*(idfs-diff)
      
        
        # normalized tfidf vector
        length = math.sqrt(length)
        weights[i,:] = weights[i,:]/length

        # print(",\t".join([str(round(i,4)) for i in weights[i,:]]))
    return files,weights


if __name__ == "__main__":
    sizes_to_cut = [1024]
    stages = [
        [ARCHIVE_TRAIN_DATA_PATH, TRAINING_DATA_PATH],
        [ARCHIVE_VAL_DATA_PATH, VALIDATION_DATA_PATH],
        [ARCHIVE_TEST_DATA_PATH, TEST_DATA_PATH],
    ]

    for stage in stages:
        x_raw = os.path.join(stage[0], "x")
        y_raw = os.path.join(stage[0], "y")

        x = os.path.join(stage[1], "x", "img")
        y = os.path.join(stage[1], "y", "img")
        for size in sizes_to_cut:
            print(f"Cutting images to size {size}")
            cut_ims_in_directory(
                x_raw,
                x,
                (size, size),
                format="tiff",
            )
            cut_ims_in_directory(
                y_raw,
                y,
                (size, size),
                mask=True,
                preprocess=True,
                format="tiff",
            )

    print("Calculating distribution of classes in training data")
    files = os.listdir(os.path.join(TRAINING_DATA_PATH, "y", "img"))
    distribution, df = get_distribution_seg(
        os.path.join(TRAINING_DATA_PATH, "y", "img"), files
    )
    print("Distribution:")

    for key, value in distribution.items():
        print(f"{key}: {value}, {df[key]}")

    # save distribution to csv
    with open(os.path.join(TRAINING_DATA_PATH, "distribution.csv"), "w") as f:
        for key in distribution.keys():
            f.write("%s,%s,%s\n" % (key, distribution[key], df[key]))

    # get tf-idf weights
    files, weights = class_tfidf_vectorizer(TRAINING_DATA_PATH)
    print("total weights:")
    print(np.sum(weights,axis=0))
    with open(os.path.join(TRAINING_DATA_PATH, "weights.csv"), "w") as f:
        for i in range(weights.shape[0]):
            f.write(",".join([files[i]]+[str(i) for i in weights[i,:]])+"\n")

