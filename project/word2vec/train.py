
import pandas as pd
import numpy as np

from gensim.models import Word2Vec

NROWS = 1000
FILE_COUNT = 'local/word2vec/count.txt'
FILE_MODEL = 'local/word2vec/reviews.word2vec.model'
FILE_DATASET = 'project/dataset/Reviews.csv'

def getDataset(offset):
    file = open(FILE_DATASET,'r')
    return pd.read_csv(file, header=0, index_col=0, usecols=['Id', 'Text'], skiprows=range(1,offset), nrows=NROWS, encoding='utf-8')

with open(FILE_COUNT,'r') as file:
    OFFSET = int(file.read())
    file.close()

dataset = getDataset(OFFSET)

try:
    model = Word2Vec.load(FILE_MODEL)
except IOError:
    model = Word2Vec(size=40, window=2, workers=1, min_count=1)

model.train(dataset.Text)

model.save(FILE_MODEL)

with open(FILE_COUNT,'w') as file:
    file.write(str(OFFSET+NROWS))
