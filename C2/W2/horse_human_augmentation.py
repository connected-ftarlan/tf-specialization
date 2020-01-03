from tensorflow.keras.preprocessing.image import ImageDataGenerator
from C2.W1.visualize_feature_maps import plot_loss, plot_accuracy
from C1.W4.horse_human import human_horse_model, compile_fit


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
        :
    """
    train_gen = ImageDataGenerator(rescale=1 / 255, rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True,
                                   fill_mode='nearest')
    valid_gen = ImageDataGenerator(rescale=1 / 255)

    train_gen = train_gen.flow_from_directory(train_path,
                                              target_size=(300, 300),
                                              batch_size=128,
                                              class_mode='binary')
    valid_gen = valid_gen.flow_from_directory(valid_path,
                                              target_size=(300, 300),
                                              batch_size=32,
                                              class_mode='binary')

    return train_gen, valid_gen


def train_model():
    base_path = '../../Data/cats-and-dogs_reduced'
    train_path = base_path + '/train'
    valid_path = base_path + '/validation'
    train_generator, valid_generator = load_data(train_path, valid_path)
    model = human_horse_model()
    history = compile_fit(model, train_generator, valid_generator)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
    plot_loss(history)
    plot_accuracy(history)


