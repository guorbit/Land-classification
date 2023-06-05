from keras_segmentation.models.unet import vgg_unet, resnet50_unet, mobilenet_unet
from keras_segmentation.models.segnet import (
    vgg_segnet,
    resnet50_segnet,
    mobilenet_segnet,
)
from keras_segmentation.models.fcn import fcn_32_vgg, fcn_32_resnet50, fcn_32_mobilenet


MODEL_MAP = {
    "vgg_unet": {
        "model": vgg_unet,
        "image_size": (384, 384),
        "output_size": (192, 192),
    },  # image size must be 192x192
    "vgg_segnet": {
        "model": vgg_segnet,
        "image_size": (384, 384),
        "output_size": (192, 192),
    },
    "vgg_fcn_32": {
        "model": fcn_32_vgg,
        "image_size": (384, 384),
        "output_size": (384, 384),
    },
    "resnet50_unet": {
        "model": resnet50_unet,
        "image_size": (384, 384),
        "output_size": (192, 192),
    },  # image size must be 192x192
    "resnet50_segnet": {
        "model": resnet50_segnet,
        "image_size": (384, 384),
        "output_size": (192, 192),
    },
    "resnet50_fcn_32": {
        "model": fcn_32_resnet50,
        "image_size": (384, 384),
        "output_size": (384, 384),
    },
    "mobilenet_unet": {
        "model": mobilenet_unet,
        "image_size": (384, 384),
        "output_size": (384, 384),
    },  # image size must be 224x224
    "mobilenet_segnet": {
        "model": mobilenet_segnet,
        "image_size": (384, 384),
        "output_size": (384, 384),
    },
    "mobilenet_fcn_32": {
        "model": fcn_32_mobilenet,
        "image_size": (384, 384),
        "output_size": (384, 384),
    },
    "unknown": {"model": None, "image_size": (512, 512), "output_size": (512, 512)},
}
