import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D,Dense
from tensorflow.keras.models import Model


def load_data():
    """
    Loads the imdb reviews dataset into memory for training

    Returns:
        train_sentences: [str]. List of reviews. Each review is a string
        train_labels: np.array(dtype=int). Numpy array of review labels
        test_sentences: [str]. List of reviews. Each review is a string
        test_labels: np.array(dtype=int). Numpy array of review labels
    """
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, valid_data = imdb['train'], imdb['test']

    train_sentences, train_labels = __create_dataset(train_data)
    test_sentences, test_labels = __create_dataset(valid_data)

    return train_sentences, train_labels, test_sentences, test_labels


def __create_dataset(data):
    """
    Given an iterator over tuples of tensors, parses the tensors into
    sentences and their corresponding labels.

    Args:
        data: iterator of tensor tuples. Each of the form (sentence, label)

    Returns:
        sentences: np.array of strings. Each is an imdb review
        labels: np.array of ints. Each is an imdb label
    """
    sentences, labels = [], []

    for sentence, label in data:
        sentences.append(str(sentence.numpy()))
        labels.append(label.numpy())

    return sentences, np.array(labels)


def prepare_train_data(train_sentences, vocab_size, max_length):
    """
    Given a list of strings, where each string is an imdb review, returns a
    padded and tokenized sequence of each string based on the entire corpus

    Args:
        train_sentences: [str]. List of strings. Each string is an imdb review
        vocab_size: int. Max number of words to create the tokenizer with
        max_length: int. Max length of a sentence

    Returns:
        tokenizer: instance of the tokenizer trained on the training data
        padded: np.array of shape (num_sentences, max_length). Each
        row is a training example represented in the form of a
        sequence of numbers padded to `max_length`
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_sentences)
    sequences = tokenizer.texts_to_sequences(train_sentences)
    padded = pad_sequences(sequences, maxlen=max_length)

    return tokenizer, padded


def prepare_test_data(tokenizer, test_sentences, max_length):
    """
    Given a trained tokenizer on the training data and a list of test
    strings, it returns a padded and tokenized sequence of the testing data

    Args:
        tokenizer: instance of the tokenizer trained on the training corpus
        test_sentences: [str]. List of strings. Each string is an imdb review
        max_length: int. Maximum length of the sequence

    Returns:
        test_padded: np.array of shape (num_sentences, max_length). Tokenized
        and padded sequence of the testing data
    """
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences, maxlen=max_length)

    return test_padded


def word_indices(tokenizer):
    """
    Creates word index and a sequence index dictionaries

    Args:
        tokenizer: instance of the trained tokenizer object

    Returns:
        word_index: {str:int}. Mapping each word to an index
        sequence_index: {int:str}. Mapping each index to its corresponding word
    """
    word_index = tokenizer.word_index
    sequence_index = {(value, key) for (key, value) in word_index.items()}
    return word_index, sequence_index


def imdb_model(max_length, vocab_size, emd_dim):
    """
    Defines a ML model to classify imdb movie reviews into positive and
    negative classes.

    Args:
        max_length: int. Shape of the input vector
        vocab_size: int. Number of words in the vocabulary
        emd_dim: int. Number of dimensions to encode each word with

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=max_length)
    x = Embedding(vocab_size, emd_dim)(visible)
    x = GlobalAveragePooling1D()(x)
    # x = Flatten()(x)  # can alternatively use Flatten() instead of pool
    x = Dense(6, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=visible, outputs=x)


def compile_fit(model, xtrain, ytrain, xtest, ytest, num_epochs):
    """
    Compiles and fits the model to the training set. Validates on the test set.

    Args:
        model: `Model` object, containing the model
        xtrain: np.array. Padded and tokenized training sentences
        ytrain: np.array. Labels for training sentences
        xtest: np.array. Padded and tokenized test sentences
        ytest: np.array. Labels for testing sentences
        num_epochs: int. Number of epochs to train the model

    Returns:
        history: `History` object, containing the history of model training
    """
    # compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # fitting the model to the training data
    history = model.fit(xtrain, ytrain, epochs=num_epochs, validation_data=(
        xtest, ytest))

    return history


def train_model():
    # setting variables
    vocab_size = 10000
    max_length = 120
    emb_dim = 16
    epochs = 10

    # preparing the data
    train_sentences, train_labels, test_sentences, test_labels = load_data()
    tokenizer, train_pad = prepare_train_data(train_sentences, vocab_size, max_length)
    test_pad = prepare_test_data(tokenizer, test_sentences, max_length)

    # training the model
    model = imdb_model(max_length, vocab_size, emb_dim)
    history = compile_fit(model, train_pad, train_labels, test_pad,
                          test_labels, epochs)

    return model, history


if __name__ == '__main__':
    model, history = train_model()
