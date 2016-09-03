# @author : Himanshu Choudhary 
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

import numpy as np
import nltk.data
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn

class LexiconSentimentAnalyzer():
	def __init__(self):
		self.sentence_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
		
	def __tokenize(self,text):
		return word_tokenize(text)

	def analyzeSentence(self,sentence):
		score=0.0
		count=0
		tokens = self.__tokenize(sentence)
		tagged = pos_tag(tokens)
		for i in range(0,len(tagged)):
			if 'JJ' in tagged[i][1] and list(swn.senti_synsets(tagged[i][0],'a')):
				synset = list(swn.senti_synsets(tagged[i][0],'a'))
				fraction=1
			elif 'VB' in tagged[i][1] and list(swn.senti_synsets(tagged[i][0],'v')):
				synset = list(swn.senti_synsets(tagged[i][0],'v'))
				fraction=1
			# elif 'RB' in tagged[i][1] and list(swn.senti_synsets(tagged[i][0],'r')):
			# 	synset = list(swn.senti_synsets(tagged[i][0],'r'))
			# 	fraction=1
			# elif 'NN' in tagged[i][1] and list(swn.senti_synsets(tagged[i][0],'n')):
			# 	synset = list(swn.senti_synsets(tagged[i][0],'n'))
			# 	fraction=1
			else:
				synset = list()
			pscore=0.0
			nscore=0.0
			if synset:
				pscore = np.mean([x.pos_score()*fraction for x in synset])
				nscore = np.mean([x.neg_score()*fraction for x in synset])

			if pscore+nscore>0:
				score+= pscore if pscore>=nscore else -1*nscore
				count+=1
		if count:
			return score/count
		else:
			return score/len(tokens)

	def analyzeSentenceWithoutTags(self,sentence):
		score=0.0
		count=0
		tokens = self.__tokenize(sentence)
		for token in tokens:
			pscore=0.0
			nscore=0.0
			synset = list(swn.senti_synsets(token))
			if synset:
				pscore = np.mean([x.pos_score() for x in synset])
				nscore = np.mean([x.neg_score() for x in synset])
			if pscore+nscore>0:
				score+= pscore if pscore>=nscore else -1*nscore
				count+=1
		if count:
			return score/count
		else:
			return score/len(tokens)

	def analyzeText(self,text):
		scores = []
		sentences = self.sentence_segmenter.tokenize(text)		
		for sentence in sentences:
			scores.append(self.analyzeSentence(sentence))
		if scores:
			return np.mean(scores)
		else:
			return 0

