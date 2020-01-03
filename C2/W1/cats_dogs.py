from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from C1.W3.MyCallback import MyCallback


def load_data(train_path, valid_path):
    """
    Given the path to the folders containing the training and validation
    data, returns ImageDataGenerator objects for the training and validation
    datasets.

    Args:
        train_path: str, path to the folder containing the training data
        valid_path: str, path to the folder containing the validation data

    Returns:
        train_generator: ImageDataGenerator, image generator for training data
        valid_generator: ImageDataGenerator, image generator for validation data
    """
    # rescaling the images to [0,1] range
    train_imagegen = ImageDataGenerator(rescale=1 / 255)
    valid_imagegen = ImageDataGenerator(rescale=1 / 255)

    # setting the image generator parameters
    train_generator = train_imagegen.flow_from_directory(train_path,
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='binary')
    valid_generator = valid_imagegen.flow_from_directory(valid_path,
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='binary')

    return train_generator, valid_generator


def cats_dogs_model():
    """
    Defines a convolutional neural network model to classify horses and humans

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=(150, 150, 3))
    x = Conv2D(16, (3, 3), activation='relu')(visible)
    x = MaxPool2D()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=visible, outputs=output)


def compile_fit(model, train_generator, valid_generator):
    """
    Compiles and fits the model on the training set. Validates on the
    validation set.

    Args:
        model: `Model` object, containing the model
        train_generator: ImageDataGenerator, image generator for training data
        valid_generator: ImageDataGenerator, image generator for validation data

    Returns:
        history: `History` object, containing the history of model training
    """
    # compiling the model
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',
                  metrics=['acc'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='acc', min_delta=0.0001,
                               patience=5, restore_best_weights=True)
    mycall = MyCallback()

    # fitting the model to the training data. Validating on the validation set
    history = model.fit_generator(train_generator, steps_per_epoch=100,
                                  epochs=20, verbose=1, validation_steps=50,
                                  validation_data=valid_generator,
                                  callbacks=[early_stop, mycall])

    return history


def train_model():
    base_path = '../../Data/cats-and-dogs_reduced'
    train_path = base_path + '/train'
    valid_path = base_path + '/validation'
    train_generator, valid_generator = load_data(train_path, valid_path)
    model = cats_dogs_model()
    history = compile_fit(model, train_generator, valid_generator)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
