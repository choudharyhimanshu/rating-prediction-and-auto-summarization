# @author : Himanshu Choudhary 
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from sklearn import metrics
import matplotlib.pyplot as plt
from ..classifiers.LexiconClassifier import LexiconSentimentAnalyzer
from .. import Utils

data = Utils.getDataset(50,random=1)
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

# print(Y_act,file=Utils.FILE_OUT)
# print(Y_pred,file=Utils.FILE_OUT)
# print(scores,file=Utils.FILE_OUT)
print(metrics.accuracy_score(Y_act, Y_pred),file=Utils.FILE_OUT)

plt.figure(1)
plt.hist(Y_act,alpha=0.5,facecolor='red',label='Actual')
plt.hist(Y_pred,alpha=0.5,facecolor='green',label='Predicted')
plt.legend(loc=2)
plt.savefig(Utils.LOCAL_DIR+'histo.png')

plt.figure(2)
plt.plot(X_act,Y_act,'ro',label='Actual')
plt.plot(X_pred,Y_pred,'go',label='Predicted')
plt.legend(loc=1)
plt.ylim(0,7)
plt.savefig(Utils.LOCAL_DIR+'graph.png')

diff=[]
diff_abs=[]
for i in range(len(Y_act)):
	diff.append(Y_pred[i]-Y_act[i])
	diff_abs.append(abs(Y_pred[i]-Y_act[i]))

plt.figure(3)
plt.hist(diff,alpha=1)
plt.title('Difference in Predicted and Actual rating')
plt.savefig(Utils.LOCAL_DIR+'histoErr.png')

# print(diff,file=Utils.FILE_OUT)
# print(diff_abs,file=Utils.FILE_OUT)
print("Matched : %f" % (diff_abs.count(0)/len(diff_abs)), file=Utils.FILE_OUT)
print("With Diff 1 : %f" % (diff_abs.count(1)/len(diff_abs)), file=Utils.FILE_OUT)
print("With Diff >1 : %f" % ((diff_abs.count(2)+diff_abs.count(3)+diff_abs.count(4))/len(diff_abs)), file=Utils.FILE_OUT)
print("\n",file=Utils.FILE_OUT)
