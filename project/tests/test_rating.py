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

X_act = data.index.values
Y_act = data['Score'].tolist()
X_pred = []
Y_pred = []
scores = []

for index,row in data.iterrows():
	score = lex_analyzer.analyzeText(row['Text'])
	X_pred.append(index)
	scores.append(score)

for score in scores:
	Y_pred.append(Utils.scaleSentimentScoreToRating(score))

print(Y_act,file=Utils.FILE_OUT)
print(Y_pred,file=Utils.FILE_OUT)
print(scores,file=Utils.FILE_OUT)
print(metrics.accuracy_score(Y_act, Y_pred),file=Utils.FILE_OUT)

plt.hist(Y_act,alpha=0.5,facecolor='red',label='Actual')
plt.hist(Y_pred,alpha=0.5,facecolor='green',label='Predicted')
plt.legend(loc=2)
plt.savefig(Utils.LOCAL_DIR+'plot.png')

notify2.init('NLP')
n = notify2.Notification('NLP', 'Lexicon Test Execution completed.')
n.show()
