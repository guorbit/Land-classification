from shape import read_images,IMAGE_SIZE,NUM_CLASSES
from keras_segmentation.models.unet import vgg_unet, resnet50_unet, mobilenet_unet
from keras_segmentation.models.segnet import vgg_segnet, resnet50_segnet, mobilenet_segnet
from keras_segmentation.models.fcn import fcn_32_vgg, fcn_32_resnet50 , fcn_32_mobilenet
from constants import MODEL_NAME
import os

models = {
    "vgg_unet_1":vgg_unet, # image size must be 192x192
    "vgg_segnet_1":vgg_segnet,
    "vgg_fcn_32_1":fcn_32_vgg,
    "resnet50_unet_1":resnet50_unet, # image size must be 192x192
    "resnet50_segnet_1":resnet50_segnet, 
    "resnet50_fcn_32_1":fcn_32_resnet50,
    "mobilenet_unet_1":mobilenet_unet, # image size must be 224x224
    "mobilenet_segnet_1":mobilenet_segnet,
    "mobilenet_fcn_32_1":fcn_32_mobilenet
}

def create_model():
    model = models[MODEL_NAME](n_classes=NUM_CLASSES,input_height=IMAGE_SIZE[0],input_width=IMAGE_SIZE[1])
    return model

def train_model(model,images,masks):
    model.train(
        train_images =  images,
        train_annotations = masks,
        checkpoints_path = os.path.join("checkpoints",MODEL_NAME)
         , epochs=20
    )
    return model



if __name__ == "__main__":

    print("Training model "+MODEL_NAME)
    image_path = "archive_resized/train/x/"
    mask_path = "archive_resized/train/y/"

    model = create_model()
    model = train_model(model,image_path,mask_path)
    model.save(MODEL_NAME+".h5")

