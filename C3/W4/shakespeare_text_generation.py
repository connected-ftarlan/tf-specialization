from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as k

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def load_data(filename):
    try:
        with open(filename) as f:
            data = f.read().lower().split('\n')
    except FileNotFoundError as e:
        print(e)
        pass
    return data


def create_dataset(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    data_sequence = tokenizer.texts_to_sequences(data)
    x, y = [], []
    for sequence in data_sequence:
        for i in range(1, len(sequence)):
            x.append(sequence[:i])
            y.append(sequence[i])
    return tokenizer, pad_sequences(x), k.utils.to_categorical(y)


def shakespeare_model(vocab_size, embedding_dim, max_sequence_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(vocab_size//2, activation='relu', kernel_regularizer=l2()))
    model.add(Dense(vocab_size, activation='softmax'))
    return model


def compile_fit(model, x, y, epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x, y, epochs=epochs, verbose=1)
    return history


def train():
    filepath = '../../Data/shakespeare_sonnets.txt'
    data = load_data(filepath)
    tokenizer, x, y = create_dataset(data)
    vocab_size = len(tokenizer.word_index) + 1
    sequence_lenth = len(x[0, :])
    embedding_dim = 100
    epochs = 100
    model = shakespeare_model(vocab_size, embedding_dim, sequence_lenth)
    history = compile_fit(model, x, y, epochs)


if __name__ == '__main__':
    train()
