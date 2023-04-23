import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input,Flatten, Dense, Dropout,ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, Concatenate, ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ThresholdedReLU, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add, Multiply, Average, Maximum, Minimum, Subtract, Dot, ZeroPadding2D, UpSampling2D, Reshape, Permute, Cropping2D, Cropping1D, Lambda, Add
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from utilities.segmentation_utils.flowreader import FlowGenerator
from constants import NUM_CLASSES, TRAINING_DATA_PATH, TEST_DATA_PATH


class ModelGenerator():
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.1/" \
                     "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
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
    base_model = None
    levels = None
    loss_fn = None

    def __init__(self,encoder,decoder, input_shape, n_classes):
        '''
        Initializes the model generator object

        Parameters
        ----------
        encoder (string): Encoder to be used
        decoder (string): Decoder to be used
        input_shape (tuple): Input shape of the model
        n_classes (int): Number of classes

        Returns
        -------
        None
        '''
        self.name = encoder+"_"+decoder
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.n_classes = n_classes
        print("------------------")
        print("Initialized with classes: ", self.n_classes)

    class accuracy_drop_callback(keras.callbacks.Callback):
        previous_loss = None
        def on_epoch_end(self, epoch, logs={}):
            if not self.previous_loss is None and logs.get('loss') > self.previous_loss:
                print("Stopping training as loss has gotten worse")
                self.model.stop_training = True
            else:
                self.previous_loss = logs.get('loss')

    def summary(self):
        '''
        Prints the summary of the model

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        return self.model.summary()

    def fit(self, training_args, tuning_args, reader_args,val_reader_args, dataset_size = None, transfer_learning = False,enabled_blocks = None):
        '''
        Starts the training of the model

        Parameters
        ----------
        *args: Arguments to be passed to the fit function
        **kwargs: Keyword arguments to be passed to the fit function

        Returns
        -------
        None
        '''
        self.model.learning_rate = 0.0001
        enabled = []
        if transfer_learning:
            self.base_model.trainable = False
        else:
            self.base_model.trainable = True
            if enabled_blocks is None:
                enabled_blocks = [True] * len(self.base_model.layers)


        print("Enabled blocks: ", enabled_blocks)
        self.compile(self.loss_fn)
        print(self.summary())
        generator = FlowGenerator(
            **reader_args
        )
        train_generator = generator.get_generator()
        val_generator = FlowGenerator(
            **val_reader_args
        )
        val_generator = val_generator.get_generator()

        if dataset_size is None:
            dataset_size = generator.get_dataset_size()
            training_args["steps_per_epoch"] = dataset_size // training_args["batch_size"]
            tuning_args["steps_per_epoch"] = dataset_size // tuning_args["batch_size"]
            print("Dataset size: ", dataset_size)
            print("Steps per epoch: ", training_args["steps_per_epoch"])

        self.model.fit(train_generator,**training_args,validation_data = val_generator, validation_steps = 50, callbacks=[self.accuracy_drop_callback()])
        if transfer_learning:
            counter = 0
            for i in reversed(self.levels):
                if enabled_blocks[counter]:
                    print("Tuning block: ", len(self.levels) - counter)
                    i.trainable = True
                    self.model.learning_rate = 0.000000001 * (10 ** -counter)
                    self.model.compile(loss = self.loss_fn, optimizer = 'adam', metrics = ["accuracy"])
                    tuning_generator = FlowGenerator(
                        **reader_args
                    )
                 
                    val_generator = FlowGenerator(
                        **val_reader_args
                    )
                    val_generator = val_generator.get_generator()
                    tuning_generator = tuning_generator.get_generator()
                    self.model.fit(tuning_generator, **tuning_args,validation_data = val_generator, validation_steps = 50, callbacks=[self.accuracy_drop_callback()])
                counter += 1

    

    def predict(self):
        '''
        Predicts the output of the model

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.model.predict()

    def evaluate(self):
        '''
        Evaluates the model

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        '''
        self.model.evaluate()

    def compile(self,loss_fn, optimizer=tf.optimizers.SGD(momentum=0.9), metrics=["accuracy"]):
        '''
        Compiles the model

        Parameters
        ----------
        loss_fn (function): Loss function to be used
        optimizer (str): Optimizer to be used
        metrics (list): List of metrics to be used

        Returns
        -------
        None
        '''
        self.loss_fn = loss_fn
        self.model.compile(optimizer = optimizer,loss = loss_fn, metrics = metrics)

    def save(self, path):
        '''
        Saves the model

        Parameters
        ----------
        path (str): Path to save the model

        Returns
        -------
        None
        '''
        self.model.save(path)

    def output_shape(self):
        '''
        Returns the output shape of the model

        Parameters
        ----------
        None

        Returns
        -------
        tuple: Output shape of the model
        '''
        return self.model.output_shape


class VGG16_UNET(ModelGenerator):
   
    def __init__(self, input_shape, n_classes):
        '''
        Initializes a VGG16 Unet Segmentation Class

        Parameters
        ----------
        input_shape (tuple): Input shape of the image
        n_classes (int): Number of classes to be segmented

        Returns
        -------
        None
        '''
        super().__init__("vgg16", "unet", input_shape, n_classes)


    def create_model(self, load_weights = False):
        '''
        Initializes a VGG16 Unet Segmentation model

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        MERGE_AXIS = -1
            
        img_input = Input(shape=self.input_shape)
        x = tf.keras.applications.vgg16.preprocess_input(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block1_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(0.05)(x)
        f1 = x
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING, dilation_rate = 2)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=self.IMAGE_ORDERING, dilation_rate = 2)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block2_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(0.05)(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=self.IMAGE_ORDERING, dilation_rate = 4)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=self.IMAGE_ORDERING, dilation_rate = 4)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=self.IMAGE_ORDERING, dilation_rate = 4)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block3_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(0.025)(x)
        f3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=self.IMAGE_ORDERING, dilation_rate = 8)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=self.IMAGE_ORDERING, dilation_rate = 8)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=self.IMAGE_ORDERING, dilation_rate = 8)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block4_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(0.0125)(x)
        f4 = x
        
        vgg = Model(img_input, x)

        if load_weights:   
            vgg.load_weights(self.VGG_Weights_path,by_name=True,skip_mismatch=True)

        self.base_model = vgg
        #vgg.learning_rate = 0.001
        self.levels = [f1, f2, f3, f4]
        
        o = f4

        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Dropout(0.4))(o)

        o = (Conv2DTranspose(512,(2, 2), strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Dropout(0.3))(o)

        o = (Conv2DTranspose(256,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Dropout(0.2))(o)

        o = (Conv2DTranspose (128,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Dropout(0.1))(o)

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
    
    def get_base_model(self):
        return self.base_model



