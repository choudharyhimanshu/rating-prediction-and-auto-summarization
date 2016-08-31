# @author : Himanshu Choudhary 
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from sklearn import metrics
import matplotlib.pyplot as plt
import notify2
from ..classifiers.LexiconClassifier import LexiconSentimentAnalyzer
from .. import Utils

data = Utils.getDataset(10,random=1)
lex_analyzer = LexiconSentimentAnalyzer()

notify2.init('NLP')
n = notify2.Notification('NLP', 'Yayy!! You are ready to go :D.')
n.show()

print('Yayy!! You are ready to go :D.')
