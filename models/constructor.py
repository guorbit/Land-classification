import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input,Flatten, Dense, Dropout,ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, Concatenate, ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ThresholdedReLU, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from constants import NUM_CLASSES


class ModelGenerator():
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.1/" \
                     "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
    IMAGE_ORDERING = 'channels_last'
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
        print("------------------")
        print("Initialized with classes: ", self.n_classes)

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

    def save(self, path):
        self.model.save(path)

    def output_shape(self):
        return self.model.output_shape


class VGG16_UNET(ModelGenerator):

    def __init__(self, input_shape, n_classes):
        super().__init__("vgg16", "unet", input_shape, n_classes)


    def create_model(self):
            
        img_input = Input(shape=self.input_shape)
        x = tf.keras.applications.vgg19.preprocess_input(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(
            img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=self.IMAGE_ORDERING)(x)
        f1 = x
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=self.IMAGE_ORDERING)(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=self.IMAGE_ORDERING)(x)
        f3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=self.IMAGE_ORDERING)(x)
        f4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=self.IMAGE_ORDERING)(x)
        f5 = x

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)
        
        vgg = Model(img_input, x)
        
        vgg.load_weights(self.VGG_Weights_path,by_name=True,skip_mismatch=True)
        #vgg.learning_rate = 0.001
        levels = [f1, f2, f3, f4, f5]
        MERGE_AXIS = -1
        o = f4

        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = Conv2D(self.n_classes, (3, 3), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)
        o_shape = Model(img_input, o).output_shape  
        o = (Reshape((o_shape[1]*o_shape[2], -1)))(o)  
        # o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)

        
        model = Model(img_input, o)

        model.outputWidth = o_shape[2]
        model.outputHeight = o_shape[1]
        #model.learning_rate = 0.001
        self.model = model

class two_layer_model(ModelGenerator):
    '''
        This class is used to generate a model with two layers
    '''
    def __init__(self, input_shape, n_classes):
        super().__init__("vgg16", "unet", input_shape, n_classes)
        
    
    def create_model(self):
        img_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        # encoder

        # Block 1
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(img_input)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=self.IMAGE_ORDERING)(x)
        f1 = x

        # Block 2
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=self.IMAGE_ORDERING)(x)
        f2 = x

        # x = Flatten(name='flatten')(x)
        # x = Dense(256, activation='relu', name='fc1')(x)
        # x = Dense(256, activation='relu', name='fc2')(x)
        # x = Dense(256, activation='softmax', name='predictions')(x)

        #decoder
        MERGE_AXIS = -1
        o = f2

        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

        o = Conv2D(self.n_classes, (3, 3), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)

        o_shape = Model(img_input, o).output_shape
        o = (Reshape((o_shape[1]*o_shape[2], -1)))(o)
        
        o = (Activation('softmax'))(o)

        model = Model(img_input, o)

        model.outputWidth = o_shape[2]
        model.outputHeight = o_shape[1]
        
        self.model = model


