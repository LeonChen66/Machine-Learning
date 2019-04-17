import pandas as pd
import io
import sys
import numpy as np
# solve garbled
from gensim.models import word2vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

def read_SST2(filename):
    df = pd.read_table(filename)
    return df['sentence'],df['label']

def Tfid(X,max_features=10000,ngram=(1,3)):
    tvec1 = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    tvec1.fit(X)
    # X_train_tfidf = tvec1.transform(X)
    return tvec1

def word2Vec():
    word2vec_file = "Pre_trained_model/GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    return model


def GloVe(file='Pre_trained_model/glove.6B.50d.txt'):
    embeddings_index = {}  # skip information on first line
    fin = open(file, encoding='utf-8')
    for line in fin:
        items = line.replace('\r', '').replace('\n', '').split(' ')
        if len(items) < 10:
            continue
        word = items[0]

        vect = np.array([float(i) for i in items[1:]])
        embeddings_index[word] = vect

    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM=50):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def tokenizer_X(X):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    return tokenizer


def get_word_vec(X, tokenizer, maxlen=49):
    sequences = tokenizer.texts_to_sequences(X)
    data = pad_sequences(sequences,maxlen=maxlen)
    return data
    # data = list()
    # for sentence in sentence_list:
    #     # get a sentence
    #     sentence_vec = []
    #     words = text_to_word_sequence(sentence)
    #     # print(words)
    #     for word in words[:maxlen]:
    #         try:
    #             # print(word,word_vecs[word].shape)
    #             sentence_vec.append(word_vecs[word])
    #         except:
    #             pass
    #     data.append(sentence_vec)
    # data = pad_sequences(data)
    # # print(data.shape)       # add a sentence vector
    # return data



def main():
    X_train, y_train = read_SST2('SST2/train.tsv')
    X_val, y_val = read_SST2('SST2/dev.tsv')
    print(X_val)
    # X_train
    # tvec = Tfid(X_train)
    # X_train_tvec = tvec.transform(X_train)
    # print(X_train_tvec.shape)
    # embedded_index = GloVe()
    tokenizer = tokenizer_X(X_train)
    
    # embedding_matrix = get_embedding_matrix(
    #     tokenizer.word_index, embedded_index)
    # X_train_vec = get_word_vec(X_train, tokenizer)
    # X_val_vec = get_word_vec(X_val, tokenizer)
    # print(X_train_vec,X_val_vec)

if __name__ == "__main__":
    main()
