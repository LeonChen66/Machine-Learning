import numpy as np
seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, InputLayer
from keras.layers import Input, Dense, concatenate, Activation, Bidirectional,LSTM
from keras.models import Model
from keras.layers import Embedding
from keras.layers.merge import Concatenate
import matplotlib.pyplot as plt
from nlp_preprocess import *


def buildNNmodel(input_dim = 10000):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


def CNNmodel(word_index, embedding_matrix, EMBEDDING_DIM=50, MAX_SEQUENCE_LENGTH=49):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid',
                           activation='relu', strides=1)(embedded_sequences)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid',
                            activation='relu', strides=1)(embedded_sequences)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid',
                             activation='relu', strides=1)(embedded_sequences)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

    merged = Dense(512, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)
    model = Model(inputs=[sequence_input], outputs=[output])
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # model.summary()
    return model


def static_CNN(word_index, embedding_matrix, embedding_dim=50,
                filter_sizes = (3, 8),
                num_filters = 10,
                dropout_prob = (0.6, 0.8),
                hidden_dims = 50):

    model_input = Input(shape=(49,))
    z = Embedding(len(word_index)+1, embedding_dim, weights=[embedding_matrix],
                  input_length=49, name="embedding", trainable=False)(model_input)
    z = Dropout(dropout_prob[0])(z)
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters,
                    kernel_size=sz,
                    padding="valid",
                    activation="relu",
                    strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy",
                optimizer="adam", metrics=["accuracy"])
    return model


def Bi_RNN(word_index, embedding_matrix, embedding_dim=50):
    model = Sequential()
    model.add(InputLayer(input_shape=(49,)))
    model.add(Embedding(len(word_index)+1, embedding_dim, weights=[embedding_matrix],
                        input_length=49, name="embedding", trainable=False))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def plot_loss(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # X_train, y_train = read_SST2('SST2/train.tsv')
    # X_validation, y_validation = read_SST2('SST2/dev.tsv')
    # X_test, y_test = read_SST2('SST2/test.tsv')
    # tvec = Tfid(X_train, max_features=50000)
    # X_train_tvec = tvec.transform(X_train)
    # X_validation_tvec = tvec.transform(X_validation)
    # X_test_tvec = tvec.transform(X_test)

    X_train, y_train = read_SST2('SST2/train.tsv')
    X_val, y_val = read_SST2('SST2/dev.tsv')
    embedded_index = GloVe()
    tokenizer = tokenizer_X(X_train)
    embedding_matrix = get_embedding_matrix(
        tokenizer.word_index, embedded_index)
    X_train_vec = get_word_vec(X_train, tokenizer)
    X_val_vec = get_word_vec(X_val, tokenizer)
    model = CNNmodel(tokenizer.word_index, embedding_matrix)
    history = model.fit(X_train_vec, y_train, batch_size=32, epochs=5,
            validation_data=(X_val_vec, y_val))
    plot_loss(history)
    # scores = model.evaluate(X_validation_tvec, y_validation, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
