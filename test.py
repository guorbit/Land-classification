from shape import read_images, label_maps
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
if __name__ == '__main__':
    images,masks = read_images("archive/test/")
    model = load_model("vgg_unet_1.h5")
    
    prediction = model.predict(images)
    prediction = prediction.reshape(images.shape[0],32,32,7)
    for k,n in enumerate(prediction):
        f, axarr = plt.subplots(1,2)
        first_prediction = np.argmax(n,axis=2)
        pred_rgb = np.zeros((32,32,3))
        for i in range(32):
            for j in range(32):
                pred_rgb[i,j,:] = label_maps[first_prediction[i,j]]

        
        axarr[0].imshow(images[k])
        axarr[1].imshow(pred_rgb)

        plt.show()
