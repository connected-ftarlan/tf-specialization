import os
import random
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from C1.W3.MyCallback import MyCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy


def __setup_data():
    """
    Parses the data in appropriate folder structure for downstream feeding
    into the model.

    Returns:
        root_path: str. Path to the directory containing the dataset
    """
    root_path = os.path.abspath(os.path.join(__file__,
                                             '../../../Data/cats-dogs_full'))

    # source directories
    cat_source = os.path.join(root_path, 'cats_dogs-dataset/PetImages/Cat')
    dog_source = os.path.join(root_path, 'cats_dogs-dataset/PetImages/Dog')

    # destination directory path
    train_path = os.path.join(root_path, 'train')
    train_cat = os.path.join(train_path, 'cats')
    train_dog = os.path.join(train_path, 'dogs')

    valid_path = os.path.join(root_path, 'validation')
    valid_cat = os.path.join(valid_path, 'cats')
    valid_dog = os.path.join(valid_path, 'dogs')

    # creating folders structure to easily load data
    __create_directory(root_path)

    # splitting the training and validation data into their correct folders
    num_cat_pics = len(os.listdir(cat_source))
    num_dog_pics = len(os.listdir(dog_source))
    num_cat_train = len(os.listdir(train_cat))
    num_cat_valid = len(os.listdir(valid_cat))
    num_dog_train = len(os.listdir(train_dog))
    num_dog_valid = len(os.listdir(valid_dog))

    if num_cat_train + num_cat_valid + 1 != num_cat_pics or \
            num_dog_train + num_dog_valid + 1 != num_dog_pics:
        split_ratio = 0.9

        # clearing the content of the directories from previous runs
        __clear_dir(train_cat)
        __clear_dir(train_dog)
        __clear_dir(valid_cat)
        __clear_dir(valid_dog)

        __split_data(cat_source, train_cat, valid_cat, split_ratio)
        __split_data(dog_source, train_dog, valid_dog, split_ratio)

    return root_path


def __create_directory(root_path):
    """
    Given the path to the mother directory, creates the folder structure for
    training and validation on cats and dogs images

    Args:
        root_path: str. Path to the mother folder
    """
    try:
        train_path = os.path.join(root_path, 'train')
        valid_path = os.path.join(root_path, 'validation')

        os.makedirs(os.path.join(train_path, 'cats'))
        os.makedirs(os.path.join(train_path, 'dogs'))
        os.makedirs(os.path.join(valid_path, 'cats'))
        os.makedirs(os.path.join(valid_path, 'dogs'))
    except OSError:
        pass


def __clear_dir(path):
    """
    Deletes all the files in the directory. Does not remove any
    subdirectories of the path.

    Args:
        path: str. Path to the folder to empty its content
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isdir(file_path):
            os.remove(file_path)


def __split_data(source, train_destination, valid_destination, split_ratio):
    """
    Splits the images in the directory `path` into training and validation
    sets based on the `split_ratio` given.

    Args:
        source: str. Path to the directory containing images
        train_destination: str. Path to the directory to copy training data to
        valid_destination: str. Path to directory to copy validation data to
        split_ratio: float. In range [0, 1], indicating the train/test split
    """
    # creating a list of all images in directory passed in
    all_images = [f for f in os.listdir(source)
                  if os.path.getsize(os.path.join(source, f)) > 0]

    # computing the number of images that should go in each directory
    num_data = len(all_images)
    num_train = int(num_data * split_ratio)

    # shuffling the images and assigning to train and test sets
    shuffled_images = random.sample(all_images, num_data)
    train_set = shuffled_images[:num_train]
    valid_set = shuffled_images[num_train:]

    # copying the images over to their corresponding directories
    __copy_over(source, train_destination, train_set)
    __copy_over(source, valid_destination, valid_set)


def __copy_over(source, destination, files_to_copy):
    """
    Given the paths to the directory containing the images nad directory to
    copy images to, copies the files indicated in `files_to_copy` over from
    source to destination.

    Args:
        source: str. Path to the directory containing the images
        destination: str. Path to the directory to copy files to
        files_to_copy: [str], file names of files to copy
    """
    for f in files_to_copy:
        current = os.path.join(source, f)
        copy(current, destination)


def load_data(train_path, valid_path):
    """
    Given a path to directory containing the training and validation data,
    it returns ImageDataGenerator objects for the training and validation
    datasets.

    Args:
        train_path: str. Path to the folder containing training data
        valid_path: str. Path to the folder containing validation data

    Returns:
        train_gen: ImageDataGenerator, image generator for training data
        valid_gen: ImageDataGenerator, image generator for validation data
    """
    train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True,
                                       fill_mode='nearest')
    train_gen = train_datagen.flow_from_directory(train_path,
                                                      target_size=(150, 150),
                                                      batch_size=512,
                                                      class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1/255)
    valid_gen = validation_datagen.flow_from_directory(valid_path,
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary')

    return train_gen, valid_gen


def cat_dogs_model():
    """
    Defines a convolutional neural network model to classify cats and dogs

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=(150, 150, 3))
    x = Conv2D(16, (3, 3), activation='relu')(visible)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
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
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4),
                  metrics=['acc'])

    # callbacks to stop training earlier than planned
    early_stop = EarlyStopping(monitor='acc', min_delta=0.0001,
                               patience=5, restore_best_weights=True)
    mycall = MyCallback()

    # fitting the model to the training data. Validating on the validation set
    history = model.fit_generator(train_generator, epochs=20, verbose=1,
                                  validation_data=valid_generator)

    return history


def train_model():
    root_path = __setup_data()
    train_path = os.path.join(root_path, 'train')
    valid_path = os.path.join(root_path, 'validation')
    train_generator, valid_generator = load_data(train_path, valid_path)
    model = cat_dogs_model()
    history = compile_fit(model, train_generator, valid_generator)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
