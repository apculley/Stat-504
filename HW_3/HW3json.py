# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:18:59 2015

@author: thyme
"""

import json,requests
u = 'http://www.webpages.uidaho.edu/~stevel/504/reviews_Musical_Instruments.json'
r=json.loads(requests.get(u).text)
r.status_code
r.json()['Overall']

dict=r.json()

r.text

#webpages.uidaho.edu/erichs/music.json

import urllib
urllib.urlretrieve('http://www.webpages.uidaho.edu/erichs/music.json', '/home/thyme/StatAnalytics/music.json')
import pandas as pd
with open('/git/data/hw3/music.json', 'rU') as f:
    data = [json.loads(row) for row in f]
    data2 = pd.DataFrame(data)

music_best_worst = data2[(data2.overall==5) | (data2.overall==1)]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(music_best_worst, music_best_worst.overall, random_state=1234567)


#Pull out review text, vectorize
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
train_dtm = vect.fit_transform(X_train.reviewText)
test_dtm = vect.transform(X_test.reviewText)

#roc_auc_score will get confused if y_test contains fives and ones, so you will need to create a new object that contains ones and zeros instead.
import numpy as np
y_test_binary = np.where(y_test==5, 1, 0)
#Null Model

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_class = nb.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob_nb = nb.predict_proba(X_test)[:, 1]

from sklearn.cross_validation import cross_val_score
nbscores = cross_val_score(nb, X, felrec, cv=10, scoring='accuracy')
nbscores
nbscores.mean()


# Try make blobs for noisy rep
    
