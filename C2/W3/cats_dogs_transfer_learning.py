from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
import os
from C2.W2.cats_dogs_augmentation import load_data, compile_fit


def cat_dog_model():
    """
    Defines a model on the top of the Inception v3 network by adding a few
    fully connected layers on the top of Inception.

    Returns:
        `Model` object, containing the model
    """
    # loading the inception v3 model from the .h5 file
    inception_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inception_model = InceptionV3(input_shape=(150, 150, 3),
                                  include_top=False, weights=None)
    inception_model.load_weights(inception_path)

    # setting the parameters in previous layers to non trainable
    for layer in inception_model.layers:
        layer.trainable = False

    # designating a layer in the inception network as the last layer
    last_layer = inception_model.get_layer('mixed7')

    # defining our model on the top of inception v3
    x = Flatten()(last_layer.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inception_model.input, outputs=x)


def train_model():
    root_path = '../../Data/cats-and-dogs_reduced'
    train_path = os.path.join(root_path, 'train')
    valid_path = os.path.join(root_path, 'validation')
    train_gen, valid_gen = load_data(train_path, valid_path)
    model = cat_dog_model()
    history = compile_fit(model, train_gen, valid_gen)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
