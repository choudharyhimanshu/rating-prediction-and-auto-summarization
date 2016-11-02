
import pandas as pd
import codecs
import re

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

NROWS = 1000
FILE_COUNT = 'local/word2vec/count.txt'
FILE_VOCAB = 'local/word2vec/reviews.vocab'
FILE_DATASET = 'project/dataset/Reviews.csv'

def getDataset(offset):
    return pd.read_csv(FILE_DATASET, header=0, index_col=0, usecols=['Id', 'Text'], skiprows=range(1,offset), nrows=NROWS, encoding='utf-8')

with codecs.open(FILE_COUNT,'r',encoding='utf-8') as file:
    OFFSET = int(file.read())
    file.close()

with codecs.open(FILE_VOCAB,'r',encoding='utf-8') as file:
    vocab = file.read().splitlines()
    file.close()

dataset = getDataset(OFFSET)

vectorizer = CountVectorizer(binary=True,tokenizer=word_tokenize)
vectorizer.fit(dataset.Text)
new_vocab = vectorizer.get_feature_names()

with codecs.open(FILE_VOCAB,'a',encoding='utf-8') as file:
    for word in new_vocab:
        word = re.sub("\d+", "", word).strip()
        if word and word not in vocab:
            file.write(word+'\n')
            print(word)
    file.close()

with open(FILE_COUNT,'w') as file:
    file.write(str(OFFSET+NROWS))

