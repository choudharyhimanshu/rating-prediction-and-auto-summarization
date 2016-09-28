
from __future__ import print_function
from ..classifiers.LSTMClassifier import LSTMReviewAnalyzer
from .. import Utils
import numpy as np
from sklearn.cross_validation import train_test_split

np.random.seed(7)

dataset = Utils.getDataset(100,random=0)
reviews = dataset['Text'].values
ratings = dataset['Score'].values

train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(reviews,ratings,test_size=.2)

analyzer = LSTMReviewAnalyzer()

analyzer.train(train_reviews,train_ratings)

score = analyzer.evaluate(test_reviews,test_ratings)
print("# of reviews in dataset : {:d}".format(len(dataset)),file=Utils.FILE_OUT)
print("# of reviews in training set : {:d}".format(len(train_reviews)),file=Utils.FILE_OUT)
print("# of reviews in testing set : {:d}".format(len(test_reviews)),file=Utils.FILE_OUT)
print("Accuracy : {:f}".format(score[1]),file=Utils.FILE_OUT)
print("\n",file=Utils.FILE_OUT)
