
import numpy as np
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

class LSTMReviewAnalyzer():

    def __init__(self,vocab,max_review_len=100):
        self.max_review_len = max_review_len
        self.vocab = vocab
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab) + 1, 128, input_length=max_review_len, dropout=0.2))
        self.model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
        self.model.add(Dense(1, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    def __indexize(self,token):
        try:
            return self.vocab[self.vocab['token'] == token].index[0]
        except IndexError:
            return 0

    def __pad(self,sequence):
        # return pad_sequences(sequence,self.max_review_len)
        return sequence[0:self.max_review_len] if len(sequence) > self.max_review_len else np.concatenate(
            (sequence, np.zeros(self.max_review_len - len(sequence), dtype=np.int32)))

    def __weight(self,rating):
        return 2 if rating in [-1,-.5] else 1

    def train(self,reviews,ratings):
        reviews = map(word_tokenize,reviews)
        for i in range(len(reviews)):
            reviews[i] = self.__pad(map(self.__indexize,reviews[i]))
        reviews = np.array(reviews)
        ratings = (ratings-3)/2.0
        # sample_weight = np.asarray(map(lambda x : 5/(2*x + 3), ratings))
        # weights = np.asarray(map(self.__weight, ratings))
        self.model.fit(reviews, ratings, nb_epoch=30, batch_size=32)

    def summary(self):
        return self.model.summary()

    def evaluate(self,reviews,ratings):
        reviews = map(word_tokenize, reviews)
        for i in range(len(reviews)):
            reviews[i] = self.__pad(map(self.__indexize,reviews[i]))
        reviews = np.array(reviews)
        ratings = (ratings-3)/2.0
        return self.model.evaluate(reviews, ratings)

    def predict(self,review):
        review = word_tokenize(review)
        review = self.__pad(map(self.__indexize,review))
        review = np.array([review])
        return self.model.predict(review)[0][0]
