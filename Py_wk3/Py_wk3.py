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

data_all = pd.read_csv('C:\\Machine_Learning\\amazon_baby.csv')
#remove all 3* review
data = data_all[data_all.rating!=3]
#show the statistics of rating

#only extract product babytrend
data = data[data.name == 'Baby Trend Diaper Champ']

plt.hist(data.rating)
plt.show()
plt.close()
data_label = data.rating >=4
#show the statistics of label
plt.hist(data_label)
plt.show()
plt.close()
#change review to string
#data['review'] = data['review'].astype('str')
#split string
#data['review'] = data['review'].ftr_apply(lambda x: x.split())
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
#associate word with index, built two dictionary word to index and index to word
ftr_selected = dict(zip(selected_words,selected_index))
ftr_selected_rev = dict(zip(selected_index,selected_words))
#extract ftr matrix ???
ftr_train = ftrmat[:,selected_index]
#ftr_train = ftrmat
#split the data into trian and test 0.8
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(ftr_train,data_label,test_size=0.2,random_state=0)
#build the classfier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
classifier = LogisticRegression()
classifier.fit(xtrain,ytrain)
score = f1_score(ytest,classifier.predict(xtest))
yprob = classifier.decision_function(xtest)
precision,recall, _ = precision_recall_curve(ytest,yprob)
print(score)
print(precision)
print(recall)
fig = plt.figure(figsize=(6,6))
fig.canvas.set_window_title("Amazon product classifier")
plt.title('Precision-Recall Curves')
plt.xlabel('Precision')
plt.ylabel('Recall')
label = 'Logistic regression F1 score = {:.3f}'.format(score)
#label object has to be defined first otherwise plt.legend will cause a warning
plt.plot(recall,precision,label=label)
plt.legend(loc='best')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()
plt.close()
#calculate the frequency of each word in select words
ftr_train_freq = ftr_train.toarray().sum(axis=0).tolist()
#get the least and most frequent word
minword = ftr_selected_rev[selected_index[ftr_train_freq.index(min(ftr_train_freq))]]
maxword = ftr_selected_rev[selected_index[ftr_train_freq.index(max(ftr_train_freq))]]
print("Least frequent word is ",minword)
print("Most frequent word is ",maxword)
#get the weight from the model
coefficent = classifier.coef_[0].tolist()
minword = ftr_selected_rev[selected_index[coefficent.index(min(coefficent))]]
maxword = ftr_selected_rev[selected_index[coefficent.index(max(coefficent))]]
print("Most negative word is ",minword)
print("Most positive word is ",maxword)



