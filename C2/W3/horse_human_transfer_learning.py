from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from C1.W3.MyCallback import MyCallback
import os


def load_data(train_dir, validation_dir):
    """
    Given a path to directory containing the training and validation data,
    it returns ImageDataGenerator objects for the training and validation
    datasets.

    Args:
        train_dir: str. Path to the folder containing training data
        validation_dir: str. Path to the folder containing validation data

    Returns:
        train_generator: ImageDataGenerator, image generator for training data
        valid_generator: ImageDataGenerator, image generator for validation data
    """
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    valid_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(
                                                        150, 150))

    return train_generator, valid_generator


def horse_human_model():
    # loading the inception v3 model from the .h5 file
    inception_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inception_model = InceptionV3(input_shape=(150, 150, 3),
                                  include_top=False, weights=None)
    inception_model.load_weights(inception_path)

    # setting the parameters in previous layers to non trainable
    for layer in inception_model.layers:
        layer.trainable = False

    # designating a layer in the inception network as the last layer
    last_layer = inception_model.get_layer('mixed7')

    # defining our model on the top of inception v3
    x = Flatten()(last_layer.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inception_model.input, outputs=x)


def compile_fit(model, train_generator, valid_generator):
    # compiling the model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy',
                  metrics=['acc'])

    # fitting the model to the training data, validating on the validation set
    history = model.fit_generator(train_generator, validation_data=valid_generator,
                        steps_per_epoch=100, epochs=100, validation_steps=50,
                        verbose=2, callbacks=[MyCallback()])

    return history


def train_model():
    root_path = '../../Data/horse-or-human'
    train_dir = os.path.join(root_path, 'Training')
    valid_dir = os.path.join(root_path, 'Validation')
    train_gen, valid_gen = load_data(train_dir, valid_dir)
    model = horse_human_model()
    history = compile_fit(model, train_gen, valid_gen)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
