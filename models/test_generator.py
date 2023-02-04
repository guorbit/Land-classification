from constructor import ModelGenerator, VGG16_UNET
from loss_constructor import LossConstructor
import numpy as np
if __name__ == "__main__":
    model = VGG16_UNET((512,512,3), 2)
    print(model.name)
    model.create_model()
    print(model.summary())
    loss_fn = LossConstructor(7,np.array([1,1,1,1,1,1,1])).weighted_cce
    model.compile(loss_fn)
