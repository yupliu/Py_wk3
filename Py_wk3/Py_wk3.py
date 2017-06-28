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
ftrmat = count_vect.fit_transform(data['review'].values.astype('U'))
#xtrain = count_vect.fit_transform(data['review'].tolist())
ftrname = count_vect.get_feature_names()
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']




