from shape import read_images, label_maps, IMAGE_SIZE
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
if __name__ == '__main__':
    

    model = load_model("vgg_unet_1.h5")
    images,masks = read_images("archive/test/")

    prediction = model.predict(images)
    print(prediction.shape)
    prediction = prediction.reshape(images.shape[0],IMAGE_SIZE[0]//2,IMAGE_SIZE[0]//2,7)
    for k,n in enumerate(prediction):
        f, axarr = plt.subplots(1,2)
        first_prediction = np.argmax(n,axis=2)
        pred_rgb = np.zeros((IMAGE_SIZE[0]//2,IMAGE_SIZE[0]//2,3))
        for i in range(IMAGE_SIZE[0]//2):
            for j in range(IMAGE_SIZE[0]//2):
                pred_rgb[i,j,:] = label_maps[first_prediction[i,j]]

        
        axarr[0].imshow(images[k])
        axarr[1].imshow(pred_rgb)

        plt.show()
