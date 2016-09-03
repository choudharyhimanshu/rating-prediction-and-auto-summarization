# @author : Himanshu Choudhary 
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from sklearn import metrics
import matplotlib.pyplot as plt
from ..classifiers.LexiconClassifier import LexiconSentimentAnalyzer
from .. import Utils

data = Utils.getDataset(10,random=1)
lex_analyzer = LexiconSentimentAnalyzer()

print('Yayy!! You are ready to go :D.')
