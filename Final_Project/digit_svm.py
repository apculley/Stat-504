# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:06:52 2015

@author: thyme
"""

#try decision_function_shape in A3
import pandas as pd
digits=pd.read_csv('/home/thyme/StatAnalytics/smalldigitstrain.csv')
pixels=digits.iloc[:,2:]
import numpy as np
pixels = np.asarray(pixels, 'float32')
pixels = pixels/255.0
label=digits['label'].tolist()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pixels, label, random_state=1234567)

from sklearn import svm
from sklearn.cross_validation import cross_val_score
clf = svm.SVC(probability=True)
svcgscores = cross_val_score(clf, X, label, cv=10, scoring='accuracy')
svcgscores
svcgscores.mean()


clf.fit(X_train, y_train)
y_prob=clf.predict_proba(X_test)
clf.predict_proba(X_test).shape

y_pred_class = clf.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)