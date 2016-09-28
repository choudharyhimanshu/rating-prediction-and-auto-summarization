
import pandas as pd
import numpy as np
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

class LSTMReviewAnalyzer():
    def __init__(self):
        # self.vocab = None
        pass

    def __indexize(self,token):
        try:
            return self.vocab[self.vocab['token'] == token].index[0]
        except IndexError:
            return len(self.vocab)

    def __pad(self,sequence):
        # return pad_sequences(sequence,self.max_review_len)
        return sequence[0:self.max_review_len] if len(sequence) > self.max_review_len else np.concatenate(
            (sequence, np.zeros(self.max_review_len - len(sequence), dtype=np.int32)))

    def train(self,reviews,ratings,max_review_len=100):
        self.max_review_len = max_review_len
        reviews = map(word_tokenize,reviews)
        tokens = []
        for review in reviews:
            tokens.extend(review)
        self.vocab = pd.DataFrame(list(set(tokens)), columns=['token'])
        for i in range(len(reviews)):
            reviews[i] = self.__pad(map(self.__indexize,reviews[i]))
        reviews = np.array(reviews)
        ratings = (ratings-3)/2.0
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab)+1, 128, input_length=max_review_len, dropout=0.2))
        self.model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.fit(reviews, ratings, nb_epoch=10, batch_size=32)

    def evaluate(self,reviews,ratings):
        reviews = map(word_tokenize, reviews)
        for i in range(len(reviews)):
            reviews[i] = self.__pad(map(self.__indexize,reviews[i]))
        reviews = np.array(reviews)
        ratings = (ratings-3)/2.0
        return self.model.evaluate(reviews, ratings)
