import tensorflow as tf
import numpy as np
from models.loss_constructor import Semantic_loss_functions

# Define input images
img1 = np.random.rand(16, 256, 256, 7)
img2 = np.random.rand(16, 256, 256, 7)

# Check input images for invalid values
tf.debugging.check_numerics(img1, "img1 contains invalid values")
tf.debugging.check_numerics(img2, "img2 contains invalid values")

# Calculate MS-SSIM
ms_ssim = tf.image.ssim_multiscale(
    img1, img2, max_val=1.0
)


loss_object = Semantic_loss_functions()
loss_fn = loss_object.ssim_loss

returned = loss_fn(img1, img2)


# Print result
print("ssim local:",ms_ssim)
print("ssim loss:",returned)

