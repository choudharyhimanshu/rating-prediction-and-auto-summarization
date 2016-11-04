
from __future__ import print_function
from .. import Utils
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

np.random.seed(7)
MAX_REVIEW_LEN = 10

def indexize(token):
    try:
        return vocab[vocab['token'] == token].index[0]
    except IndexError:
        return 0

def pad(sequence):
    # return pad_sequences(sequence,self.max_review_len)
    return sequence[0:MAX_REVIEW_LEN] if len(sequence) > MAX_REVIEW_LEN else np.concatenate(
        (sequence, np.zeros(MAX_REVIEW_LEN - len(sequence), dtype=np.int32)))

def make_binary(rating):
    if rating > 3:
        return 1
    return 0

dataset = Utils.getDataset(5000,random=1)
dataset = dataset[dataset['Score'] != 3]

reviews = dataset['Summary'].values
ratings = np.array(dataset['Score'].map(make_binary).values)
ratings = to_categorical(ratings)

reviews = np.array(map(Utils.filterStopwords,reviews))
reviews = map(word_tokenize,reviews)

tokens = []
for review in reviews:
    tokens.extend(review)
vocab = pd.DataFrame(list(set(tokens)), columns=['token'])
vocab.index += 1

for i in range(len(reviews)):
    reviews[i] = pad(map(indexize,reviews[i]))
reviews = np.array(reviews)

train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(reviews,ratings,test_size=.2)

# print(vocab)
# print(train_reviews)
# print(train_ratings)

model = Sequential()
model.add(Embedding(len(vocab) + 1, 256, input_length=MAX_REVIEW_LEN, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(train_reviews,train_ratings,nb_epoch=10,batch_size=32)

preds = model.predict(test_reviews)
for i in range(len(preds)):
    print(preds[i],test_ratings[i],file=Utils.FILE_OUT)
# for i in range(len(preds)):
#     print(preds[i].argmax(),test_ratings[i].argmax(),file=Utils.FILE_OUT)

score = model.evaluate(test_reviews,test_ratings)
print("Vocab size : {:d}".format(len(vocab)),file=Utils.FILE_OUT)
print("# of reviews in dataset : {:d}".format(len(dataset)),file=Utils.FILE_OUT)
print("# of reviews in training set : {:d}".format(len(train_reviews)),file=Utils.FILE_OUT)
print("# of reviews in testing set : {:d}".format(len(test_reviews)),file=Utils.FILE_OUT)
print("Accuracy : {:f}".format(score[1]),file=Utils.FILE_OUT)
print("\n",file=Utils.FILE_OUT)
