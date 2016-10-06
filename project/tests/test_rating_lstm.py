
from __future__ import print_function
from ..classifiers.LSTMClassifier import LSTMReviewAnalyzer
from .. import Utils
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split

np.random.seed(7)

dataset = Utils.getDataset(100,random=0)
reviews = dataset['Text'].values
ratings = dataset['Score']

# reviews = np.array(map(Utils.filterStopwords,reviews))

tokenized_reviews = map(word_tokenize,reviews)
tokens = []
for review in tokenized_reviews:
    tokens.extend(review)
vocab = pd.DataFrame(list(set(tokens)), columns=['token'])
vocab.index += 1
train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(reviews,ratings,test_size=.2)

indices = test_ratings.index
test_ratings = test_ratings.values
train_ratings = train_ratings.values

analyzer = LSTMReviewAnalyzer(vocab)
analyzer.train(train_reviews,train_ratings)

score = analyzer.evaluate(test_reviews,test_ratings)
print("Vocab size : {:d}".format(len(vocab)),file=Utils.FILE_OUT)
print("# of reviews in dataset : {:d}".format(len(dataset)),file=Utils.FILE_OUT)
print("# of reviews in training set : {:d}".format(len(train_reviews)),file=Utils.FILE_OUT)
print("# of reviews in testing set : {:d}".format(len(test_reviews)),file=Utils.FILE_OUT)
print("Accuracy : {:f}".format(score[1]),file=Utils.FILE_OUT)
print("\n",file=Utils.FILE_OUT)
