from shape import read_images,IMAGE_SIZE,NUM_CLASSES
from keras_segmentation.models.unet import vgg_unet 
import os
def create_model():
    model = vgg_unet(n_classes=NUM_CLASSES ,  input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1])
    return model

def train_model(model,images,masks):
    model.train(
        train_images =  images,
        train_annotations = masks,
        checkpoints_path = os.path.join("checkpoints","vgg_unet_1")
         , epochs=20
    )
    return model



if __name__ == "__main__":
    image_path = "archive_resized/train/x/"
    mask_path = "archive_resized/train/y/"
    model = create_model()
    model = train_model(model,image_path,mask_path)
    model.save("vgg_unet_1.h5")

