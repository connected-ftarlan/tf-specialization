import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def nn_model():
    """
    Defines a neural network model. The model is a simple one-neuron neural
    network (ie. linear regression)

    Returns
    -------
    model -- keras object containing the model
    """
    model = Sequential()
    model.add(Dense(1, input_shape=[1]))
    return model


def compile_fit(x_train, y_train):
    """
    Compiles and fits the models to the training data passed in.
    Calls the nn_model function above to create the model

    Parameters
    ----------
    x_train -- numpy array, containing floats
    y_train -- numpy array, containing floats

    Returns
    -------
    model -- keras object, containing the object
    history -- keras object, containing the training history of the model on
    the training dataset passe in to the function
    """
    # defining the model
    model = nn_model()

    # compiling the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # fitting the model to the training data
    history = model.fit(x_train, y_train, epochs=500)

    return model, history


if __name__ == '__main__':
    x_train = np.arange(0, 10, dtype=float)
    y_train = np.arange(0.5, 5.5, 0.5, dtype=float)

    model, history = compile_fit(x_train, y_train)
    print(model.predict([7.0]))
