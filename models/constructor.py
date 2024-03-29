"""
Custom model generator class
"""

import tensorflow as tf

import itertools
from tensorflow import keras
from keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    Input,
    Dropout,
    ZeroPadding2D,
    Concatenate,
    ReLU,
    ZeroPadding2D,
    UpSampling2D,
)

from keras.models import Model
from utilities.segmentation_utils.flowreader import FlowGenerator
from constants import NUM_CLASSES, TRAINING_DATA_PATH, TEST_DATA_PATH


@DeprecationWarning
class ModelGenerator_old:
    pretrained_url = (
        "https://github.com/fchollet/deep-learning-models/"
        "releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
        pretrained_url.split("/")[-1], pretrained_url
    )
    IMAGE_ORDERING = "channels_last"
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

    def __init__(self, encoder, decoder, input_shape, n_classes):
        """
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
        """
        self.name = encoder + "_" + decoder
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.n_classes = n_classes
        print("------------------")
        print("Initialized with classes: ", self.n_classes)

    class accuracy_drop_callback(keras.callbacks.Callback):
        previous_loss = None

        def on_epoch_end(self, epoch, logs={}):
            if not self.previous_loss is None and logs.get("loss") > self.previous_loss:
                print("Stopping training as loss has gotten worse")
                self.model.stop_training = True
            else:
                self.previous_loss = logs.get("loss")

    def summary(self):
        """
        Prints the summary of the model

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        return self.model.summary()

    def fit(
        self,
        training_args,
        tuning_args,
        reader_args,
        val_reader_args,
        dataset_size=None,
        transfer_learning=False,
        enabled_blocks=None,
    ):
        """
        Starts the training of the model

        Parameters
        ----------
        *args: Arguments to be passed to the fit function
        **kwargs: Keyword arguments to be passed to the fit function

        Returns
        -------
        None
        """
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
        generator = FlowGenerator(**reader_args)
        train_generator = generator.get_generator()
        val_generator = FlowGenerator(**val_reader_args)
        val_generator = val_generator.get_generator()

        if dataset_size is None:
            dataset_size = generator.get_dataset_size()
            training_args["steps_per_epoch"] = (
                dataset_size // training_args["batch_size"]
            )
            tuning_args["steps_per_epoch"] = dataset_size // tuning_args["batch_size"]
            print("Dataset size: ", dataset_size)
            print("Steps per epoch: ", training_args["steps_per_epoch"])

        self.model.fit(
            train_generator,
            **training_args,
            validation_data=val_generator,
            validation_steps=50,
            callbacks=[self.accuracy_drop_callback()],
        )
        if transfer_learning:
            counter = 0
            for i in reversed(self.levels):
                if enabled_blocks[counter]:
                    print("Tuning block: ", len(self.levels) - counter)
                    i.trainable = True
                    self.model.learning_rate = 0.000000001 * (10**-counter)
                    self.model.compile(
                        loss=self.loss_fn,
                        optimizer=self.optimizer,
                        metrics=["accuracy"],
                    )
                    tuning_generator = FlowGenerator(**reader_args)

                    val_generator = FlowGenerator(**val_reader_args)
                    val_generator = val_generator.get_generator()
                    tuning_generator = tuning_generator.get_generator()
                    self.model.fit(
                        tuning_generator,
                        **tuning_args,
                        validation_data=val_generator,
                        validation_steps=50,
                        callbacks=[self.accuracy_drop_callback()],
                    )
                counter += 1

    def predict(self):
        """
        Predicts the output of the model

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model.predict()

    def evaluate(self):
        """
        Evaluates the model

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.model.evaluate()

    def compile(
        self, loss_fn, optimizer=tf.optimizers.SGD(momentum=0.8), metrics=["accuracy"]
    ):
        """
        Compiles the model

        Parameters
        ----------
        loss_fn (function): Loss function to be used
        optimizer (str): Optimizer to be used
        metrics (list): List of metrics to be used

        Returns
        -------
        None
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def save(self, path):
        """
        Saves the model

        Parameters
        ----------
        path (str): Path to save the model

        Returns
        -------
        None
        """
        self.model.save(path)

    def output_shape(self):
        """
        Returns the output shape of the model

        Parameters
        ----------
        None

        Returns
        -------
        tuple: Output shape of the model
        """
        return self.model.output_shape


class ModelGenerator(Model):
    """
    Custom model class with custom training loop
    """

    name = None
    n_classes = None
    base_model = None
    levels = None
    loss_fn = None
    optimizer = None
    acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")
    loss_tracker = keras.metrics.Mean(name="loss")
    eval_acc_metric = keras.metrics.CategoricalAccuracy(name="val_accuracy")
    eval_loss_tracker = keras.metrics.Mean(name="val_loss")
    eval_recall = keras.metrics.Recall(name="val_recall")
    eval_precision = keras.metrics.Precision(name="val_precision")

    backup_logs = None
    # metrics = None

    def __init__(self, name, *args, **kwargs):
        super(ModelGenerator, self).__init__(*args, **kwargs)
        self.name = name

    def get_backup_logs(self):
        """
        returns the eval metric values from the model
        """
        return self.backup_logs

    def compile(self, *args, **kwargs):
        """
        Compiles the model
        """
        self.loss_fn = kwargs["loss"]
        self.optimizer = kwargs["optimizer"]
        kwargs["metrics"] = kwargs["metrics"] + [
            self.acc_metric,
            self.loss_tracker,
            self.eval_acc_metric,
            self.eval_loss_tracker,
            self.eval_recall,
            self.eval_precision,
        ]
        super(ModelGenerator, self).compile(*args, **kwargs)

    def save(self, path):
        """
        saves the model
        """
        super(ModelGenerator, self).save(path)

    @tf.function
    def train_step(self, data):
        """
        Custom training step, capable of handling weights
        """
        if len(data) == 3:
            x, y, w = data
        else:
            x, y = data
            w = None
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss

            if w is None:
                loss = self.compiled_loss(y, y_pred)
            else:
                loss = self.compiled_loss(y, y_pred, w)
            # loss = tf.reduce_mean(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return self.loss_tracker.result(), self.acc_metric.result()

    @tf.function
    def eval_step(self, data):
        """
        Custom evaluation step
        """
        x, y = data
        y = tf.convert_to_tensor(y)

        y_pred = self(x, training=False)
        tf.convert_to_tensor(y_pred)
        loss = self.compiled_loss(y, y_pred)
        loss = tf.reduce_mean(loss)

        self.eval_loss_tracker.update_state(loss)
        self.eval_acc_metric.update_state(y, y_pred)
        self.eval_recall.update_state(y, y_pred)
        self.eval_precision.update_state(y, y_pred)

        return self.eval_loss_tracker.result(), self.eval_acc_metric.result()

    def train(
        self,
        dataset,
        epochs=1,
        batch_size=32,
        learning_rate=1e-2,
        steps_per_epoch=512,
        validation_dataset=None,
        validation_steps=50,
        callbacks=[],
        enable_tensorboard=False,
    ):
        """
        Custom training loop
        """
        self.optimizer.learning_rate = learning_rate
        self.val_data = validation_dataset
        logs = {}
        metrics = [
            self.loss_tracker,
            self.acc_metric,
            self.eval_loss_tracker,
            self.eval_acc_metric,
            self.eval_recall,
            self.eval_precision,
        ]
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "verbose": 1})

        for callback in callbacks:  # on train begin callbacks
            callback.on_train_begin()
        dataset.set_mini_batch_size(batch_size)
        for epoch in range(epochs):
            for callback in callbacks:  # on epoch begin callbacks
                callback.on_epoch_begin(epoch)

            tf.print(f"\nEpoch: {epoch+1}/{epochs}")

            pbar = tf.keras.utils.Progbar(
                target=steps_per_epoch, stateful_metrics=["time_to_complete"]
            )

            for dataset_idx, data in enumerate(
                itertools.islice(dataset, steps_per_epoch)
            ):
                # for batch_idx, mini_batch in enumerate(data):

                loss, accuracy = self.train_step(data)
                pbar.update(
                    dataset_idx + 1, values=[("loss", loss), ("accuracy", accuracy)]
                )
                for callback in callbacks:  # on batch end callbacks
                    callback.on_train_batch_end(dataset_idx)

            pbar = tf.keras.utils.Progbar(target=validation_steps)
            if not validation_dataset is None:
                validation_dataset.set_mini_batch_size(batch_size)
                tf.print("Performing validation")
                validation_cycle = itertools.cycle(validation_dataset)
                for dataset_idx, data in enumerate(
                    itertools.islice(validation_cycle, validation_steps)
                ):
                    loss, accuracy = self.eval_step(data)
                    pbar.update(
                        dataset_idx + 1,
                        values=[
                            ("val_loss", loss),
                            ("val_accuracy", accuracy),
                            ("val_recall", self.eval_recall.result()),
                            ("val_precision", self.eval_precision.result()),
                        ],
                    )

            for metric in metrics:
                if hasattr(metric, "result"):
                    logs[metric.name] = metric.result().numpy()

                    metric.reset_states()
                else:
                    logs[metric.name] = metric.numpy()
            self.backup_logs = logs.copy()
            for callback in callbacks:  # on epoch end callbacks
                callback.on_epoch_end(epoch, logs=logs)
            # dataset.on_epoch_end()
        for callback in callbacks:  # on train end callbacks
            callback.on_train_end()


class VGG16_UNET:
    """
    Custom class defining a VGG16 Unet forwad pass
    """

    pretrained_url = (
        "https://github.com/fchollet/deep-learning-models/"
        "releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
        pretrained_url.split("/")[-1], pretrained_url
    )

    def __init__(
        self,
        input_shape,
        output_shape,
        n_classes,
        load_weights=False,
        dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        """
        Initializes a VGG16 Unet Segmentation Class

        Parameters
        ----------
        input_shape (tuple): Input shape of the image
        n_classes (int): Number of classes to be segmented

        Returns
        -------
        None
        """
        self.n_classes = n_classes
        self.IMAGE_ORDERING = "channels_last"

        print("Initializing VGG16 Unet")
        self.create_model(
            input_shape=input_shape,
            output_shape=output_shape,
            load_weights=load_weights,
            dropouts=dropouts,
        )

    def create_model(
        self,
        input_shape,
        output_shape,
        load_weights=False,
        dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        """
        Initializes a VGG16 Unet Segmentation model forward pass definition

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # fmt: off
        MERGE_AXIS = -1
            
        img_input = Input(shape=input_shape)
        
        x = tf.keras.applications.vgg16.preprocess_input(img_input)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f1 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[0])(x)
        
        
        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f2 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[1])(x)
        

        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2', data_format=self.IMAGE_ORDERING, dilation_rate = 2)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3', data_format=self.IMAGE_ORDERING, dilation_rate = 2)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f3 = x
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[2])(x)
        

        # Block 4
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2', data_format=self.IMAGE_ORDERING, dilation_rate=4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3', data_format=self.IMAGE_ORDERING, dilation_rate=4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f4 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[3])(x)
        
        # Block 5 pyramid pooling block

        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', data_format=self.IMAGE_ORDERING, dilation_rate=4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', data_format=self.IMAGE_ORDERING, dilation_rate=4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f5 = x


        vgg = ModelGenerator("vgg",inputs = img_input,outputs = x)

        if load_weights:   
            vgg.load_weights(self.VGG_Weights_path,by_name=True,skip_mismatch=True)

        self.base_model = vgg
        #vgg.learning_rate = 0.001
        self.levels = [f1, f2, f3, f4]
        
        # vgg.trainable = False

        # for layer in vgg.layers[-4:None]:
        #     print("Setting layer trainable: ", layer)
        #     layer.trainable = True
        
        f5 = PyramidPoolingModule([1,2,4,8],[(1,1),(3,3),(3,3),(3,3)],num_channels=self.n_classes)(f5)

        o = f5

        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(512, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[4]))(o)

        o = (Conv2DTranspose(512,(2, 2), strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f4]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(256, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[5]))(o)

        o = (Conv2DTranspose(256,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(128, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[6]))(o)

        o = (Conv2DTranspose (128,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(64, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[7]))(o)

        o = (Conv2DTranspose (64,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(32, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(32, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[8]))(o)

        o = Conv2D(self.n_classes, (1, 1), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)
        o_shape = Model(img_input, o).output_shape  
        # o = (Reshape((o_shape[1]*o_shape[2], -1)))(o)  
        # o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)

        
        model = ModelGenerator("vgg_unet", inputs = img_input, outputs = o)
        model.levels = [f1, f2, f3, f4]
        
        model.outputWidth = o_shape[2]
        model.outputHeight = o_shape[1]
        #model.learning_rate = 0.001
        self.model = model

        # fmt: on

    def get_model(self):
        return self.model

    def get_base_model(self):
        return self.base_model


class VGG_NANO_UNET:
    """
    Custom class defining a VGG16 Unet forwad pass
    """

    pretrained_url = (
        "https://github.com/fchollet/deep-learning-models/"
        "releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
        pretrained_url.split("/")[-1], pretrained_url
    )

    def __init__(
        self,
        input_shape,
        output_shape,
        n_classes,
        load_weights=False,
        dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        """
        Initializes a VGG16 Unet Segmentation Class

        Parameters
        ----------
        input_shape (tuple): Input shape of the image
        n_classes (int): Number of classes to be segmented

        Returns
        -------
        None
        """
        self.n_classes = n_classes
        self.IMAGE_ORDERING = "channels_last"

        print("Initializing VGG16 Unet")
        self.create_model(
            input_shape=input_shape,
            output_shape=output_shape,
            load_weights=load_weights,
            dropouts=dropouts,
        )

    def create_model(
        self,
        input_shape,
        output_shape,
        load_weights=False,
        dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        """
        Initializes a VGG16 Unet Segmentation model forward pass definition

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # fmt: off
        MERGE_AXIS = -1
            
        img_input = Input(shape=input_shape)
        
        x = tf.keras.applications.vgg16.preprocess_input(img_input)
        x = UnetNanoConvBlock(64, (3, 3), padding='same', data_format=self.IMAGE_ORDERING)(x)
        f1 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[0])(x)
        
        
        # Block 2
        x = UnetNanoConvBlock(128, (3, 3), padding='same', data_format=self.IMAGE_ORDERING)(x)
        f2 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[1])(x)
        

        # Block 3
        x = UnetNanoConvBlock(256, (3, 3), padding='same', data_format=self.IMAGE_ORDERING)(x)
        f3 = x
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[2])(x)
        

        # Block 4
        x = UnetNanoConvBlock(512, (3, 3), padding='same', data_format=self.IMAGE_ORDERING)(x)
        f4 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(dropouts[3])(x)
        
        # Block 5 pyramid pooling block

        x = UnetNanoConvBlock(512, (3, 3), padding='same', data_format=self.IMAGE_ORDERING)(x)
        f5 = x


        vgg = ModelGenerator("vgg",inputs = img_input,outputs = x)

        if load_weights:   
            vgg.load_weights(self.VGG_Weights_path,by_name=True,skip_mismatch=True)

        self.base_model = vgg
        #vgg.learning_rate = 0.001
        self.levels = [f1, f2, f3, f4]
        
        # vgg.trainable = False

        # for layer in vgg.layers[-4:None]:
        #     print("Setting layer trainable: ", layer)
        #     layer.trainable = True
        
        f5 = PyramidPoolingModule([1,4,8],[(1,1),(3,3),(3,3)],num_channels=self.n_classes)(f5)

        o = f5

        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)

        o = (Dropout(dropouts[4]))(o)

        o = (UpSampling2D((2, 2),data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f4]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[5]))(o)

        o = (UpSampling2D((2, 2),data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[6]))(o)

        o = (UpSampling2D((2, 2),data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[7]))(o)

        o = (UpSampling2D((2, 2),data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(32, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(dropouts[8]))(o)

        o = Conv2D(self.n_classes, (1, 1), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)
        o_shape = Model(img_input, o).output_shape  
        # o = (Reshape((o_shape[1]*o_shape[2], -1)))(o)  
        # o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)

        
        model = ModelGenerator("vgg_nano_unet", inputs = img_input, outputs = o)
        model.levels = [f1, f2, f3, f4]
        
        model.outputWidth = o_shape[2]
        model.outputHeight = o_shape[1]
        #model.learning_rate = 0.001
        self.model = model

        # fmt: on

    def get_model(self):
        return self.model

    def get_base_model(self):
        return self.base_model


class PyramidPoolingModule(Layer):
    """
    Custom implementation of a pyramid pooling module
    """

    def __init__(
        self, pool_sizes, kernels, num_channels, data_format="channels_last", **kwargs
    ):
        super(PyramidPoolingModule, self).__init__(**kwargs)
        self.pool_sizes = pool_sizes
        self.kernels = kernels
        self.num_channels = num_channels
        self.data_format = data_format

    def build(self, input_shape):
        self.conv_layers = []
        for pool_size, kernel in zip(self.pool_sizes, self.kernels):
            self.conv_layers.append(
                Conv2D(
                    input_shape[-1],
                    kernel,
                    padding="same",
                    data_format=self.data_format,
                    dilation_rate=pool_size,
                )
            )

    def call(self, x):
        input_shape = tf.shape(x)
        h, w = input_shape[1], input_shape[2]
        pyramid_features = [x]
        for pool_size, conv_layer in zip(self.pool_sizes, self.conv_layers):
            x = tf.keras.layers.AveragePooling2D(
                pool_size=(pool_size, pool_size),
                strides=(pool_size, pool_size),
                padding="same",
                data_format=self.data_format,
            )(x)
            x = conv_layer(x)
            x = UpSampling2D(size=(pool_size, pool_size), data_format=self.data_format)(
                x
            )

            pyramid_features.append(x)
        output = Concatenate(axis=-1)(pyramid_features)
        return output

    def get_config(self):
        config = super(PyramidPoolingModule, self).get_config()
        config.update(
            {
                "pool_sizes": self.pool_sizes,
                "kernels": self.kernels,
                "num_channels": self.num_channels,
                "data_format": self.data_format,
            }
        )
        return config


#! Note the blocks below are not tested yet
class UnetNanoConvBlock(Layer):
    def __init__(self, num_filters, kernel_size,padding, data_format="channels_last", **kwargs):
        super(UnetNanoConvBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.data_format = data_format

    def build(self, input_shape):
        self.conv_layer = Conv2D(
            self.num_filters,
            self.kernel_size,
            padding=self.padding,
            data_format=self.data_format,
        )
        self.batch_norm_layer = BatchNormalization()
        self.activation_layer = ReLU()

    def call(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        config = super(UnetNanoConvBlock, self).get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "data_format": self.data_format,
            }
        )
        return config


class UnetDecoderBlock(Layer):
    def __init__(self, num_filters, kernel_size, data_format="channels_last", **kwargs):
        super(UnetDecoderBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.data_format = data_format

    def build(self, input_shape):
        self.conv_layer = Conv2DTranspose(
            self.num_filters,
            self.kernel_size,
            strides=(2, 2),
            padding="same",
            data_format=self.data_format,
        )
        self.batch_norm_layer = BatchNormalization()
        self.activation_layer = ReLU()

    def call(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        config = super(UnetDecoderBlock, self).get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "data_format": self.data_format,
            }
        )
        return config
