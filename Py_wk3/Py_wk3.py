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

data_all = pd.read_csv('d:\\ML_Learning\\amazon_baby.csv')
#remove all 3* review
data = data_all[data_all.rating!=3]
#show the statistics of rating
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
#associate word with index
ftr_selected = dict(zip(selected_words,selected_index))
#extract ftr matrix ???
ftr_train = ftrmat[:,selected_index]
#split the data into trian and test 0.8
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(ftr_train,data_label,test_size=0.2)
#Normalize the attribute value to mean=0 and sd =1
#from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler()
#scalar.fit(xtrain)
#xtrain = scalar.transform(xtrain)
#xtest = scalar.transform(xtest)
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
plt.legend(loc='best')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.plot(recall,precision,label=label)
plt.show()
plt.close()



    



