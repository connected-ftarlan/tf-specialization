import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_feature_maps(model, img):
    """
    Given a convolutional neural network model and an appropriately shaped
    image, it plots the feature maps of the convolutional and pooling layers.

    Args:
        model: keras `Model` object, containing the model to plot feature
        maps of
        img: numpy array. Image represented in the form of a numpy array of
        shape (1, size, size, num_channels)
    """
    # lets first create a list of outputs from successive layers of the model
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # now lets create another model with the input of the model passed in,
    # and outputs of each layer of the passed in model
    visualization_model = Model(inputs=model.input, outputs=successive_outputs)

    # now run the image through the network, obtaining intermediate feature maps
    successive_feature_maps = visualization_model.predict(img)

    # layer names to have them as part of our plot for readability purposes
    layer_names = [layer.name for layer in model.layers]

    # loop through the layers
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        # plot feature maps only for conv/pool layers
        if len(feature_map.shape) == 4:
            n_channels = feature_map.shape[-1]
            size = feature_map.shape[1]

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_channels))

            # looping through each filter of a layer
            for i in range(n_channels):
                # postprocessing the image to be visually palatable
                img = feature_map[0, :, :, i]
                img -= img.mean()
                img /= img.std()
                img *= 64
                img += 128
                img = np.clip(img, 0, 255).astype('uint8')
                # tile each filter into a horizontal grid
                display_grid[:, i * size: (i + 1) * size] = img

            # displaying the feature map grid
            scale = 20. / n_channels
            plt.figure(figsize=(scale * n_channels, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


def find_image(path, model):
    """
    Given a path to a folder containing images, it returns a numpy array
    representation of the image suitable for passing through the keras
    model (also passed in) to plot feature maps.

    Args:
        path: str. Path to the folder containing images
        model: keras `Model` object.

    Returns:
        img: numpy array. Representation of the image suitable for
        visualizing feature maps of the model
    """
    # string formatting for reliability
    if not path.endswith('/'):
        path += '/'

    # creating a list of all images in directory passed in
    all_images = [f for f in os.listdir(path) if f.endswith('.jpg')]

    # finding a random image from the directory passed in
    img_path = path + random.choice(all_images)
    img = load_img(img_path, target_size=(model.input_shape[1],
                                          model.input_shape[2]))

    # convert the image into its numpy representation
    img = img_to_array(img)                 # shape: (img_size, img_size, 3)
    img = img.reshape((1,) + img.shape)     # shape: (1, size, size, 3)
    return img / 255.0


def plot_feature_maps_from_random_img(model, folder_path):
    """
    randomly chooses an image from a directory containing images and plots
    the feature maps of that image as gone through the model.

    Args:
        model: keras `Model` object, containing the model to print feature
        maps of
        folder_path: str. Path to the folder containing images
    """
    img = find_image(folder_path, model)
    plot_feature_maps(model, img)


def plot_loss(history):
    """
    Plots the loss of the model as a function of training epochs

    Args:
        history: `History` object, containing the training history of the model
    """
    loss = history.history['loss']

    # plotting the validation loss if a validation set exists
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(val_loss, label='validation loss')

    plt.plot(loss, label='training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model loss vs. training epochs')
    plt.show()


def plot_accuracy(history):
    """
    Plots the accuracy history of the model as a function of training epochs

    Args:
        history: `History` object, containing the training history of the model
    """
    acc = history.history['acc']

    # plotting the validation accuracy if there exists a validation set
    if 'val_acc' in history.history:
        val_acc = history.history['val_acc']
        plt.plot(val_acc, label='validation accuracy')

    plt.plot(acc, label='training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model accuracy vs. training epochs')
    plt.show()


if __name__ == '__main__':
    from C2.W1.cats_dogs import train_model

    # def model():
    #     a = Input(shape=(150, 150, 3))
    #     b = Conv2D(4, (3, 3), activation='relu')(a)
    #     b = MaxPool2D()(b)
    #     b = Conv2D(8, (3, 3), activation='relu')(b)
    #     b = MaxPool2D()(b)
    #     b = Flatten()(b)
    #     b = Dense(1, activation='sigmoid')(b)
    #     model = Model(inputs=a, outputs=b)
    #
    #     train_imagegen = ImageDataGenerator(rescale=1 / 255)
    #     valid_imagegen = ImageDataGenerator(rescale=1 / 255)
    #
    #     train_generator = train_imagegen.flow_from_directory(path + '/train',
    #                                                      target_size=(
    #                                                      150, 150),
    #                                                      batch_size=20,
    #                                                      class_mode='binary')
    #     valid_generator = valid_imagegen.flow_from_directory(path + '/validation',
    #                                                     target_size=(150, 150),
    #                                                     batch_size=20,
    #                                                     class_mode='binary')
    #
    #     model.compile(optimizer='adam', loss='binary_crossentropy',
    #                   metrics=['acc'])
    #
    #     history = model.fit(train_generator, steps_per_epoch=100, epochs=5,
    #                         verbose=1, validation_steps=50,
    #                         validation_data=valid_generator)
    #
    #     return model, history

    path = '../../Data/cats-and-dogs_reduced/'
    model, history = train_model()

    plot_feature_maps_from_random_img(model, path+'train/cats')
    plot_loss(history)
    plot_accuracy(history)
