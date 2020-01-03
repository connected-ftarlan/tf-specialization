from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping


def load_fashion_mnist():
    """
    Loads the fashion MNIST dataset into memory. It also normalizes the input
    data to lie between 0 and 1.

    Returns:
        x_train -- (60000,28,28) numpy array, 60000 28x28 grayscale images
        y_train -- (60000,) numpy array, 60000 0-9 integer labels
        x_valid -- (10000, 28,28) numpy array, 10000 28x28 grayscale images
        y_valid -- (10000,), 10000 0-9 integer labels
    """
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
    return x_train/255, y_train, x_valid/255, y_valid


def nn_model():
    """
    Defines a NN model to learn the fashion MNIST dataset

    Returns:
        model -- keras model object, containing the model
    """
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_fit(model, x_train, y_train, x_valid, y_valid):
    """
    Compiles and fits the model to the training data. Validates on the
    validation data passed in to the function

    Args:
        model: keras sequential object -- containing the model
        x_train: numpy array (60000,28,28) -- 60000 training 28x28 images
        y_train: numpy array (60000,) -- 60000 training labels
        x_valid: numpy array (10000,28,28) -- 10000 validation 28x28 images
        y_valid: numpy array (10000,) -- 10000 validation labels

    Returns:
        history -- History object -- containing the history of model trianing
    """
    # compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.001,
                               patience=5, mode='max', verbose=1)

    # fitting the model to the training data. Validating on the validation set
    history = model.fit(x_train, y_train, epochs=50, validation_data=(
        x_valid, y_valid), callbacks=[early_stop])

    return history


def run():
    x_train, y_train, x_valid, y_valid = load_fashion_mnist()
    model = nn_model()
    history = compile_fit(model, x_train, y_train, x_valid, y_valid)

    return model, history


if __name__ == '__main__':
    model, history = run()
