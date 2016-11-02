
from __future__ import print_function
from ..classifiers.LSTMClassifier import LSTMReviewAnalyzer
from .. import Utils
import numpy as np
import pandas as pd
import logging
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import LSTM, Dense

np.random.seed(7)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# FILE_MODEL = 'local/word2vec/reviews.word2vec.model'
FILE_MODEL = 'project/dataset/glove.6B.300d.txt'

VECTOR_SIZE = 300
MAX_REVIEW_LEN = 100

def pad(tokens):
    return tokens[:MAX_REVIEW_LEN] if len(tokens)>MAX_REVIEW_LEN else \
                        np.concatenate((tokens,['' for _ in range(MAX_REVIEW_LEN-len(tokens))]))

def vectorize(tokens):
    # return np.mean([np.array(vectorizer[token]) if token in vectorizer else np.zeros(VECTOR_SIZE) for token in tokens],axis=0)
    return np.array([np.array(vectorizer[token]) if token in vectorizer else np.zeros(VECTOR_SIZE) for token in tokens])

dataset = Utils.getDataset(2000,random=1)
reviews = dataset['Text']
ratings = dataset['Score'].values

print(dataset['Score'].value_counts())

vectorizer = Word2Vec.load_word2vec_format(FILE_MODEL, binary=False)
reviews_vectors = reviews.map(Utils.filterStopwords).map(word_tokenize).map(pad).map(vectorize)
X = []
for review in reviews_vectors:
    X.append(review)
X = np.array(X)

ratings = (ratings-3)/2.0

X_train, X_test, y_train, y_test = train_test_split(X,ratings,test_size=.2)

analyzer = Sequential()
# analyzer.add(Dense(128,input_dim=VECTOR_SIZE,activation='linear'))
analyzer.add(LSTM(128, input_length= MAX_REVIEW_LEN,input_dim=VECTOR_SIZE, dropout_W=0.2, dropout_U=0.2))
analyzer.add(Dense(1, activation='tanh'))
analyzer.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

analyzer.summary()

analyzer.fit(X_train,y_train,nb_epoch=50,batch_size=64)

for i in range(len(X_test)):
    print(analyzer.predict(np.array([X_test[i]]))[0][0],y_test[i],file=Utils.FILE_OUT)

score = analyzer.evaluate(X_test,y_test)
print("# of reviews in dataset : {:d}".format(len(dataset)),file=Utils.FILE_OUT)
print("# of reviews in training set : {:d}".format(len(y_train)),file=Utils.FILE_OUT)
print("# of reviews in testing set : {:d}".format(len(y_test)),file=Utils.FILE_OUT)
print("Accuracy : {:f}".format(score[1]),file=Utils.FILE_OUT)
print("\n",file=Utils.FILE_OUT)
