import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from C3.W2.imdb_reviews import word_indices
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model


def read_data(filename):
    """
    Given the path to the csv file containing the data, it parses the data
    into a list of sentences and a corresponding list of labels.

    Args:
        filename: str. Path to the csv file containing the data

    Returns:
        sentences: [str], where each entry is a string
        labels: [str], where each entry is a label to the corresponding
        sentence in the `sentences` list
    """
    sentences, labels = [], []
    stopwords = ["a", "about", "above", "after", "again", "against", "all",
                 "am", "an", "and", "any", "are", "as", "at", "be", "because",
                 "been", "before", "being", "below", "between", "both", "but",
                 "by", "could", "did", "do", "does", "doing", "down", "during",
                 "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here",
                 "here's", "hers", "herself", "him", "himself", "his", "how",
                 "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
                 "is", "it", "it's", "its", "itself", "let's", "me", "more",
                 "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out",
                 "over", "own", "same", "she", "she'd", "she'll", "she's",
                 "should", "so", "some", "such", "than", "that", "that's",
                 "the", "their", "theirs", "them", "themselves", "then",
                 "there", "there's", "these", "they", "they'd", "they'll",
                 "they're", "they've", "this", "those", "through", "to", "too",
                 "under", "until", "up", "very", "was", "we", "we'd", "we'll",
                 "we're", "we've", "were", "what", "what's", "when", "when's",
                 "where", "where's", "which", "while", "who", "who's", "whom",
                 "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
    stop_dic = __create_stop_word_dic(stopwords)

    try:
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)

            # looping through the csv file line by line
            for line in csv_reader:
                labels.append(line[0])
                sentences.append(__remove_stop_words(line[1], stop_dic))

    except FileNotFoundError as e:
        print(e)
        pass

    return sentences, labels


def __create_stop_word_dic(stop_word_list):
    """
    Given a list of stop words, it creates a dictionary containing the same
    words. This is for better optimized runtime complexity.

    Args:
        stop_word_list: [str], where each entry is a single word

    Returns:
        stop_dic: {str:int}. Mapping each stop word to an integer (always 1)
    """
    stop_dic = {}

    for word in stop_word_list:
        if word.lower() not in stop_dic:
            stop_dic[word.lower()] = 1
    return stop_dic


def __remove_stop_words(sentence, stop_word_dic):
    """
    Given a sentence and a dictionary of stop words, it returns a sentence
    identical to the one passed in with stop words removed.

    Args:
        sentence: str. A sentence in the form of a string
        stop_word_dic: {str:int}. Dictionary mapping stop words to an integer

    Returns:
        str. The input sentence with its stop words removed
    """
    new_sentence = []

    for word in sentence.split(' '):
        if word.lower() not in stop_word_dic:
            new_sentence.append(word)
    return ' '.join(new_sentence)


def tokenize_sentences(train_sentences, valid_sentences, num_words, oov_token, max_length, pad_type):
    """
    Tokenizes the training and validation sequences based on the hyper
    parameters passed in.

    Args:
        train_sentences: [str]. List of sentences where each entry is a
        sentence in the training set
        valid_sentences: [str]. List of sentences where each entry is a
        sentence in the validation set
        num_words: int, indicating the max number of words to be tokenized
        oov_token: str. String used for out of vocabulary tokens
        max_length: int. Maximum length of the sequence of a sentence
        pad_type: str. 'pre' or 'post' indicating the type of padding

    Returns:
        tokenizer: Tokenizer object, fitted on the training sentences
        train_padded: np array, of shape (num_train_sentences, max_length)
        where each row represents a sentence in the training set
        valid_padded: np array, of shape (num_valid_sentences, max_length)
        where each row represents a sentence in the validation set
    """
    # instantiating a tokenizer instance to tokenize sentences (strings)
    tokenizer = Tokenizer(num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_sentences)

    # tokenizing the training sentences
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=pad_type)

    # tokenizing the validation sentences
    valid_sequence = tokenizer.texts_to_sequences(valid_sentences)
    valid_padded = pad_sequences(valid_sequence, maxlen=max_length, padding=pad_type)

    return tokenizer, train_padded, valid_padded


def tokenize_labels(train_labels, valid_labels):
    """
    Given a list of labels for the training set and one for the validation
    set, it returns one corresponding np array, one for training and one for
    validation sets, where each label is tokenized into an integer

    Args:
        train_labels: [str], where each entry is label corresponding to a
        sentence in the training set
        valid_labels: [str], where each entry is label corresponding to a
        sentence in the validation set

    Returns:
        train_labels_seq: np array, of shape (num_train_labels, 1),
        where each entry is the label to the corresponding sentence in the
        training set
        valid_labels_seq: np array, of shape (num_valid_labels, 1),
        where each entry is the label to the corresponding sentence in the
        validation set
    """
    # instantiating a tokenizer instance to tokenize labels (strings)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_labels)

    # tokenizing the training labels
    train_labels_seq = tokenizer.texts_to_sequences(train_labels)

    # tokenizing the validation labels
    valid_labels_seq = tokenizer.texts_to_sequences(valid_labels)

    return np.array(train_labels_seq), np.array(valid_labels_seq)


def bbc_model(max_length, vocab_size, emb_dim):
    """
    Defines a ML model to classify bbc news articles into six categories

    Args:
        max_length: int. Shape of the input vector
        vocab_size: int. Number of words in the vocabulary
        emd_dim: int. Number of dimensions to encode each word with

    Returns:
        `Model` object, containing the model
    """
    visible = Input(shape=max_length)
    x = Embedding(vocab_size, emb_dim)(visible)
    x = GlobalAveragePooling1D()(x)
    x = Dense(24, activation='relu')(x)
    x = Dense(6, activation='softmax')(x)
    return Model(inputs=visible, outputs=x)


def compile_fit(model, train_padded, train_labels, valid_padded, valid_labels, num_epochs):
    """
    Compiles and fits the model to the training set. Validates on the test set.

    Args:
        model: `Model` object, containing the model
        train_padded: np.array. Padded and tokenized training sentences
        train_labels: np.array. Labels for training sentences
        valid_padded: np.array. Padded and tokenized test sentences
        valid_labels: np.array. Labels for testing sentences
        num_epochs: int. Number of epochs to train the model

    Returns:
        history: `History` object, containing the history of model training
    """
    # compiling the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # fitting the model to the training data
    history = model.fit(train_padded, train_labels, epochs=num_epochs,
                        validation_data=(valid_padded, valid_labels))

    return history


def train_model():
    """
    Trains the model.

    Returns:
        model: `Model` instance, containing the model
        history: `History` object, containing the model training history
    """
    # setting some model hyper-parameters
    num_epochs = 30
    max_length = 120
    emb_dim = 16
    vocab_size = 1000
    pad_type = 'post'
    oov_token = '<OOV>'
    train_test_split = 0.8

    # reading the data from the csv file
    filename = '../../Data/bbc-text.csv'
    sentences, labels = read_data(filename)

    # splitting the data into training and validation sets
    train_index = int(len(sentences) * train_test_split)
    train_sentences = sentences[:train_index]
    train_labels = labels[:train_index]
    valid_sentences = sentences[train_index:]
    valid_labels = labels[train_index:]

    # tokenizing the training and validation sentences and labels
    tokenizer, x_train, x_valid = tokenize_sentences(train_sentences,
                                                     valid_sentences,
                                                     vocab_size, oov_token,
                                                     max_length, pad_type)
    y_train, y_valid = tokenize_labels(train_labels, valid_labels)

    # creating a word index and a sequence index dictionary
    word_index, seq_index = word_indices(tokenizer)

    # creating and training the model
    model = bbc_model(max_length, vocab_size, emb_dim)
    history = compile_fit(model, x_train, y_train, x_valid, y_valid, num_epochs)

    return model, history


if __name__ == '__main__':
    model, history = train_model()