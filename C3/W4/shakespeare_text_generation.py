import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def load_data(filename):
    """
    Given the path to a text file containing the dataset, it returns a list
    of strings.

    Args:
        filename: str. Path to the text file containing the data

    Returns:
        - [str]. List of strings of the data
    """
    try:
        with open(filename) as f:
            data = f.read().lower().split('\n')
    except FileNotFoundError as e:
        print(e)
        pass

    return data


def create_dataset(data):
    """
    Creates a dataset from the list of strings passed in, where each data
    points is constructed by slicing each sentence to various degrees. The
    label of each data point is the next word that appears in the sentence.

    Args:
        data: [str]. List of strings containing the data

    Returns:

    """
    tokneizer = Tokenizer()
    tokneizer.fit_on_texts(data)
    data_sequence = tokneizer.texts_to_sequences(data)
    x, y = [], []
    max_length = max([len(line) for line in data_sequence])
    print(data_sequence)
    print(len(data_sequence))
    print(len(data))

    return np.array(x), np.array(y)


def shakespeare_model():
    pass


def compile_fit():
    pass


if __name__ == '__main__':
    filepath = '../../Data/shakespeare_sonnets.txt'
    data = load_data(filepath)
    x, y = create_dataset(data)
    print(len(data))
    print(x.shape)
    print(y.shape)