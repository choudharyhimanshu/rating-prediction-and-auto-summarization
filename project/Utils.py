# @author : Himanshu Choudhary 
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

import numpy as np
from pandas import read_csv
import os.path
import project

# Globals
LOCAL_DIR = 'local/'
FILE_INP = open(LOCAL_DIR+'inp.txt','r')
FILE_OUT = open(LOCAL_DIR+'out1.txt','a')

# Range for rating from 1 to 5
DELTA_1 = [-1,-0.03]
DELTA_2 = [-0.03,-0.01]
DELTA_3 = [-0.01,0.01]
DELTA_4 = [0.01,0.03]
DELTA_5 = [0.03,1]
def scaleSentimentScoreToRating(value):
	if value < DELTA_1[1]:
		return 1
	elif value >= DELTA_2[0] and value < DELTA_2[1]:
		return 2
	elif value >= DELTA_3[0] and value <= DELTA_3[1]:
		return 3
	elif value > DELTA_4[0] and value <= DELTA_4[1]:
		return 4
	elif value > DELTA_5[0]:
		return 5

def getDataset(nrows=100,random=0):
	dataset_dir = os.path.join(os.path.dirname(project.__file__), 'dataset')
	dataset_file = os.path.join(dataset_dir, 'Reviews.csv')
	if random:
		data = read_csv(dataset_file,header=0,index_col=0,usecols=['Id','Score','Text'],nrows=50*nrows)
		rand = np.random.random_integers(0,49*nrows)
		data = data[rand:rand+nrows]
		return data
	else:
		return read_csv(dataset_file,header=0,index_col=0,usecols=['Id','Score','Text'],nrows=nrows)
