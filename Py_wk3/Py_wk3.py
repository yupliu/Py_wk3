# _*_ codeing utf-8 _*_

"""
Created on Jan 25 2017

@author: u551896
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn
except ImportError:
    pass

data = pd.read_csv('d:\\ML_Learning\\amazon_baby.csv')
#change review to string
#data['review'] = data['review'].astype('str')
#split string
#data['review'] = data['review'].apply(lambda x: x.split())
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
#extract review into bag of words and count the frequency
#ftrmat stores the count of each words
ftrmat = count_vect.fit_transform(data['review'].values.astype('U'))
#xtrain = count_vect.fit_transform(data['review'].tolist())
#ftrname stors the name of these words
ftrname = count_vect.get_feature_names()
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
selected_index = []
ftr_train = []
for word in selected_words:
    #retrieve the index of the word in ftr list
    word_index = ftrname.index(word)
    selected_index.append(word_index)    
#associate word with index
ftr_selected = dict(zip(selected_words,selected_index))
#extract ftr matrix
for word in selected_words:
    ftr_train.extend(ftrmat[ftr_selected[word]])


    



