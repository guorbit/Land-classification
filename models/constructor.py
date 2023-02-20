import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


class ModelGenerator():
    name = None
    encoder = None
    decoder = None
    input_shape = None
    output_shape = None
    n_classes = None
    model = None

    def __init__(self,encoder,decoder, input_shape, n_classes):
        self.name = encoder+"_"+decoder
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.n_classes = n_classes

    def summary(self):
        return self.model.summary()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self):
        self.model.predict()

    def evaluate(self):
        self.model.evaluate()

    def compile(self,loss_fn, optimizer="adam", metrics=["accuracy"]):
        self.model.compile(optimizer = optimizer,loss = loss_fn, metrics = metrics)


class VGG16_UNET(ModelGenerator):

    def __init__(self, input_shape, n_classes):
        super().__init__("vgg16", "unet", input_shape, n_classes)


    def create_model(self):
        inputs = Input(self.input_shape)

        vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
        
        #extracting features of the encoder blocks (forwards pass)
        s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
        s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
        s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
        s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)

        #bridge
        b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)

        #decoder
        d1 = self.decoder_block(b1, s4, 512)                     ## (64 x 64)
        d2 = self.decoder_block(d1, s3, 256)                     ## (128 x 128)
        d3 = self.decoder_block(d2, s2, 128)                     ## (256 x 256)
        d4 = self.decoder_block(d3, s1, 64)                      ## (512 x 512)

        #output
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        
        model = Model(inputs, outputs, name="VGG16_U-Net")

        self.model = model

    def conv_block(self,input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
        return x
    
    def decoder_block(self,input, skip_features, num_filters):
        x = self.conv_block(input, num_filters)
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
        x = Concatenate()([x, skip_features])
        
        return x
    

