from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,\
    Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from C1.W3.MyCallback import MyCallback
import os


def load_data(train_dir, valid_dir):
    """
    Given a path to directory containing the training and validation data,
    it returns ImageDataGenerator objects for the training and validation
    datasets.

    Args:
        train_dir: str. Path to the folder containing training data
        valid_dir: str. Path to the folder containing validation data

    Returns:
        train_generator: ImageDataGenerator, image generator for training data
        valid_generator: ImageDataGenerator, image generator for validation data
    """
    training_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2, zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = training_datagen.flow_from_directory(train_dir,
                                                   target_size=(150, 150),
                                                   batch_size=11,
                                                   class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(valid_dir,
                                                  target_size=(150, 150),
                                                  batch_size=11,
                                                  class_mode='categorical')

    return train_generator, validation_generator


def rps_model():
    """
    Defines a convolutional neural network model to classify rock papers
    scissors. The pictures are generated through GCI

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=(150, 150, 3))
    x = Conv2D(64, (3, 3), activation='relu')(visible)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    return Model(inputs=visible, outputs=x)


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
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='accuracy', min_delta=0.0001,
                               patience=5, restore_best_weights=True)

    # fitting the model to the training data. Validating on the validation set
    history = model.fit_generator(train_generator, epochs=25,
                                  steps_per_epoch=79, verbose=1,
                                  validation_data=valid_generator,
                                  validation_steps=79,
                                  callbacks=[early_stop, MyCallback()])

    return history


def train_model():
    root_dir = '../../Data/rps'
    train_dir = os.path.join(root_dir, 'rps-train')
    valid_dir = os.path.join(root_dir, 'rps-valid')
    train_generator, valid_generator = load_data(train_dir, valid_dir)
    model = rps_model()
    history = compile_fit(model, train_generator, valid_generator)
    return model, history


if __name__ == '__main__':
    model, history = train_model()

