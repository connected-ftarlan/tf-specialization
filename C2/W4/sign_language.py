import numpy as np
import csv
import math
import os
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def __read_data(filename):
    with open(filename) as data:
        X, y = [], []
        csv_reader = csv.reader(data, delimiter=',')
        line_count = 0

        # looping through the file line by line - skipping the first line
        for line in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            a = int(math.sqrt(len(line[1:])))
            X.append(np.asarray(line[1:], dtype=np.float32).reshape(a, a, 1))
            y.append(int(line[0]))

        return np.asarray(X), np.asarray(y)


def load_data(x_train, y_train, x_valid, y_valid):
    training_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2, zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow(x_train, y_train, batch_size=32)
    validation_generator = validation_datagen.flow(x_valid, y_valid,
                                                   batch_size=32)

    return train_generator, validation_generator


def sign_model():
    """
    Defines a convolutional neural network model to classify horses and humans

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=(28, 28, 1))
    x = Conv2D(64, (3, 3), activation='relu')(visible)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(25, activation='softmax')(x)
    return Model(inputs=visible, outputs=x)


def compile_fit(model, train_generator, valid_generator):
    """
    Compiles and fits the model on the training set. Validates on the
    validation set.

    Args:
        model: `Model` object, containing the model
        train_generator:
        valid_generator:
    Returns:
        history: `History` object, containing the history of model training
    """
    # compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # fitting the model to the training data, validating on the validation set
    history = model.fit_generator(train_generator, epochs=20,
                                  steps_per_epoch=int(27455/32),
                                  validation_data=valid_generator,
                                  validation_steps=int(7172/32))

    return history


def train_model():
    root_dir = '../../Data/sign-language-mnist'
    train_filename = os.path.join(root_dir, 'sign_mnist_train.csv')
    valid_filename = os.path.join(root_dir, 'sign_mnist_valid.csv')

    # reading the data off the csv file
    x_train, y_train = __read_data(train_filename)
    x_valid, y_valid = __read_data(valid_filename)

    # defining a generator to create batches of data at every epoch
    train_generator, valid_generator = load_data(x_train, y_train, x_valid,
                                                 y_valid)

    # defining the model and fitting it to the data
    model = sign_model()
    history = compile_fit(model, train_generator, valid_generator)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
