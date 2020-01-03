import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


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

        return x_train, y_train, x_test, y_test

    except FileNotFoundError as e:
        print(filename, 'not found!')

    except IOError as e:
        print('IO error occurred')


def nn_model():
    """
    Defines a deep neural network model with input size of 28x28 to classify
    10 classes

    Returns:
        model -- keras object, containing the model
    """
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_fit(model, x_train, y_train, x_valid, y_valid):
    """
    Compiles and fits the model to the training data passed in to the function

    Args:
        model: keras object, containing the model
        x_train: numpy array of shape (None, 28,28) of training data
        y_train: numpy array of shape (None,) of training labels
        x_valid: numpy array of shape (None, 28, 28) of validation data
        y_valid: numpy array of shape (None,) of validation labels

    Returns:
        history -- History object, containing the history of model training
    """
    # compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # fitting the model to the training data. Validating on the validation set
    history = model.fit(x_train, y_train, epochs=10, validation_data=(
        x_valid, y_valid), callbacks=[])

    return history


def run():
    filename = '../../Data/mnist.npz'
    x_train, y_train, x_valid, y_valid = load_data(filename)
    model = nn_model()
    history = compile_fit(model, x_train, y_train, x_valid, y_valid)
    return model, history


if __name__ == '__main__':
    model, history = run()
