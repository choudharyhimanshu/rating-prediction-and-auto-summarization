
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def remove_html(text):
    soup = BeautifulSoup(text,"html.parser")
    return soup.get_text()

LIMIT = 25000

dataset = pd.read_csv('project/dataset/Reviews.csv',header=0,index_col=0,usecols=['Id','Score','Summary','Text'],encoding='utf-8')

with_1 = dataset.loc[dataset['Score'] == 1].sample(n=LIMIT)
with_2 = dataset.loc[dataset['Score'] == 2].sample(n=LIMIT)
with_3 = dataset.loc[dataset['Score'] == 3].sample(n=LIMIT)
with_4 = dataset.loc[dataset['Score'] == 4].sample(n=LIMIT)
with_5 = dataset.loc[dataset['Score'] == 5].sample(n=LIMIT)

new_dataset = pd.concat([with_1,with_2,with_3,with_4,with_5]).sort_index()
new_dataset['Text'] = new_dataset['Text'].map(remove_html)

new_dataset.to_csv('project/dataset/Reviews_uniform_'+str(LIMIT)+'.csv',encoding='utf-8')

