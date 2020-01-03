import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, \
    Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_data(filename):
    """
    Loads the MNIST dataset into memory in the from of four numpy arrays.

    Args:
        filename: path to the .npz file storing the data. The data is stored
        in the form of a dictionary with the keys being x_train, y_train,
        x_test, y_test

    Returns:
        x_train -- numpy array of shape (60000, 28,28) of training images
        y_train -- numpy array of shape (60000,) of training labels
        x_test -- numpy array of shape (10000, 28,28) of validation images
        y_test -- numpy array of shape (10000,) of validation labels
    """
    try:
        with np.load(filename) as data:
            x_train = data['x_train'] / 255
            y_train = data['y_train']
            x_test = data['x_test'] / 255
            y_test = data['y_test']

            # reshaping the data for easier feed into the model
            x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[2], 1))
            x_test = x_test.reshape((-1, x_test.shape[1], x_test.shape[2], 1))

        return x_train, y_train, x_test, y_test

    except FileNotFoundError as e:
        print(filename, 'not found!')

    except IOError as e:
        print('IO error occurred')


def conv_model():
    """
    Defines a convolutional NN model to learn the MNIST dataset

    Returns:
        model -- keras model object, containing the model
    """
    input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model


def compile_fit(model, x_train, y_train, x_valid, y_valid):
    """
    Compiles and fits the model to the training data. Validates on the
    validation data passed in to the function

    Args:
        model: keras model object -- containing the model
        x_train: numpy array (60000,28,28,1) -- 60000 training 28x28 images
        y_train: numpy array (60000,) -- 60000 training labels
        x_valid: numpy array (10000,28,28,1) -- 10000 validation 28x28 images
        y_valid: numpy array (10000,) -- 10000 validation labels

    Returns:
        history -- History object -- containing the history of model training
    """
    # compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                               mode='max', patience=5,
                               restore_best_weights=True)

    # fitting the model to the training data. Validating on the validation set
    history = model.fit(x_train, y_train, epochs=20, validation_data=(
        x_valid, y_valid), callbacks=[early_stop])

    return history


def run():
    filename = '../../Data/mnist.npz'
    x_train, y_train, x_valid, y_valid = load_data(filename)
    model = conv_model()
    history = compile_fit(model, x_train, y_train, x_valid, y_valid)

    return model, history


if __name__ == '__main__':
    model, history = run()
