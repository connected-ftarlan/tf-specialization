from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, \
    Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist
from C1.W3.MyCallback import MyCallback


def load_data():
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
    x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[-1], 1))
    x_valid = x_valid.reshape((-1, x_valid.shape[1], x_valid.shape[-1], 1))
    return x_train /255, y_train, x_valid / 255, y_valid


def conv_model():
    """
    Defines a convolutional NN model to learn the fashion MNIST dataset

    Returns:
        model -- keras model object, containing the model
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_fit(model, x_train, y_train, x_valid, y_valid):
    """
    Compiles and fits the model to the training data. Validates on the
    validation data passed in to the function

    Args:
        model: keras sequential object -- containing the model
        x_train: numpy array (60000,28,28,1) -- 60000 training 28x28 images
        y_train: numpy array (60000,) -- 60000 training labels
        x_valid: numpy array (10000,28,28,1) -- 10000 validation 28x28 images
        y_valid: numpy array (10000,) -- 10000 validation labels

    Returns:
        history -- History object -- containing the history of model trianing
    """
    # compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                               patience=5, mode='max', verbose=1)
    mycall = MyCallback()

    # fitting the model to the training data. Validating on the validation set
    history = model.fit(x_train, y_train, epochs=50, validation_data=(
        x_valid, y_valid), callbacks=[early_stop, mycall])

    return history


def run():
    x_train, y_train, x_valid, y_valid = load_data()
    model = conv_model()
    history = compile_fit(model, x_train, y_train, x_valid, y_valid)

    return model, history


if __name__ == '__main__':
    model, history = run()
