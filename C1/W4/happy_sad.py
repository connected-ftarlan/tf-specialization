from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        threshold = 0.999
        if logs.get('acc') > threshold:
            if epoch + 1 > 1:
                print('\nTrained for {} epochs'.format(epoch+1))
            else:
                print('\nTrained for {} epoch'.format(epoch+1))
            print('Achieved validation accuracy of >{}%'.format(threshold*100))
            print('Stopping training...')
            self.model.stop_training = True


def load_data(path):
    """
    Given a path to the folder containing the training data, returns
    ImageDataGenerator object for training on the dataset.

    Args:
        path: str. path to the folder containing the data

    Returns:
        train_generator: ImageDataGenerator. Image generator for training data
    """
    train_imagegen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_imagegen.flow_from_directory(path,
                                                        target_size=(150, 150),
                                                        batch_size=16,
                                                        class_mode='binary')
    return train_generator


def happy_sad_model():
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


def compile_fit(model, train_generator):
    """
    Compiles and fits the model on the training set. Validates on the
    validation set.

    Args:
        model: `Model` object, containing the model
        train_generator: ImageDataGenerator, image generator for training data

    Returns:
        history: `History` object, containing the history of model training
    """
    # compiling the model
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy',
                  metrics=['acc'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='acc', min_delta=0.0001,
                               patience=5, restore_best_weights=True)
    mycall = MyCallback()

    # fitting the model to the training data. Validating on the validation set
    history = model.fit_generator(train_generator, steps_per_epoch=5,
                                  epochs=20, verbose=1,
                                  callbacks=[early_stop, mycall])

    return history


def train_model():
    path = '../../Data/happy-or-sad'
    train_generator = load_data(path)
    model = happy_sad_model()
    history = compile_fit(model, train_generator)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
