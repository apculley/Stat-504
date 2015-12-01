# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:11:34 2015

@author: thyme
"""

"""
digits=pd.read_csv('/home/thyme/Stat Analytics/train.csv')
smalldigits=digits.head(n=300)
smalldigits.to_csv('/home/thyme/Stat Analytics/smalldigitstrain.csv')
"""
import pandas as pd
digits=pd.read_csv('/home/thyme/Stat Analytics/smalldigitstrain.csv')

import scipy
label=digits['label'].tolist()
pixels=digits.iloc[:,2:]
pixels=scipy.sparse.csr_matrix(pixels.values)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pixels, label, random_state=1234567)

#Data visualization
row2=digits.iloc[1,:]
import collections
collections.Counter(label)
histogram = collections.Counter(label)

# Naive Bayes:
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_class = nb.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(nb, pixels, label, cv=10, scoring='accuracy')
scores
scores.mean()

#KNN:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
# search for an optimal value of K for KNN
k_range = range(1, 11)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, pixels, label, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# 10-fold cross-validation with the best KNN model
#knn = KNeighborsClassifier(n_neighbors=?)
knnscores = cross_val_score(knn, pixels, label, cv=10, scoring='accuracy')
knnscores
knnscores.mean()

#SVC Gaussian - one vs one
import numpy as np
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(label, n_iter=5, test_size=0.2, random_state=42)
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
grid = GridSearchCV(SVC, param_grid=param_grid, cv=3, scoring='accuracy')
grid.fit(pixels, label)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo',C=?, gamma=?, kernel='rbf')
from sklearn.cross_validation import cross_val_score
svcgscores = cross_val_score(clf, pixels, label, cv=10, scoring='accuracy')
svcgscores
svcgscores.mean()

#SVC Gaussian - one vs all (looks like this isn't easily available)
import numpy as np
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(label, n_iter=5, test_size=0.2, random_state=42)
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
grid = GridSearchCV(SVC(decision_function_shape='ovr'), param_grid=param_grid, cv=cv)
grid.fit(pixels, label)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovr',C=?, gamma=?, kernel='rbf')
from sklearn.cross_validation import cross_val_score
svcgscores = cross_val_score(clf, pixels, label, cv=10, scoring='accuracy')
svcgscores
svcgscores.mean()

#Decision Tree

from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)

from sklearn.cross_validation import cross_val_score
treescores = cross_val_score(treeclf, pixels, label, cv=10, scoring='accuracy')
treescores
treescores.mean()

#Bagged Decision Tree

from sklearn.ensemble import BaggingClassifier

#Find best n_estimators
potential_n = range(10, 210, 10)
Accuracy_scores = []
for n in potential_n:
    bagclass = BaggingClassifier(treeclf,n_estimators=n, random_state=1)
    scores = cross_val_score(bagclass, pixels, label, cv=5, scoring='accuracy')
    Accuracy_scores.append(scores.mean())

# plot n_estimators (x-axis) versus Accuracy (y-axis)
import matplotlib.pyplot as plt
plt.plot(potential_n, Accuracy_scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')

bagging=BaggingClassifier(treeclf,n_estimators=?, random_state=1)

from sklearn.cross_validation import cross_val_score
bagscores = cross_val_score(bagging, pixels, label, cv=10, scoring='accuracy')
bagscores
nbagscores.mean()

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()

#Find best n_estimators
potential_n = range(10, 210, 10)
Accuracy_scores = []
for n in potential_n:
    rfclass = RandomForestClassifier(n_estimators=n, random_state=1)
    scores = cross_val_score(rfclass, pixels, label, cv=5, scoring='accuracy')
    Accuracy_scores.append(scores.mean())
    
# plot n_estimators (x-axis) versus Accuracy (y-axis)
import matplotlib.pyplot as plt
plt.plot(potential_n, Accuracy_scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')

# Find best max_features
feature_range = range(1, len(feature_cols)+1)
Accuracy_scores = []

# use 10-fold cross-validation with each value of max_features
for feature in feature_range:
    rfclass = RandomForestClassifier(n_estimators=?, max_features=feature, random_state=1)
    scores = cross_val_score(rfclass, pixels, label, cv=10, scoring='accuracy')
    Accuracy_scores.append(scores.mean())

# plot max_features (x-axis) versus Accuracy (y-axis)
plt.plot(feature_range, Accuracy_scores)
plt.xlabel('max_features')
plt.ylabel('accuracy')

rfclass=RandomForestCLassifier(n_estimators=?, max_features=?, random_state=1)
from sklearn.cross_validation import cross_val_score
rfscores = cross_val_score(rfclass, pixels, label, cv=10, scoring='accuracy')
rfscores
rfscores.mean()

#Bernoulli Restricted Boltzmann Machine (RBM)

# Load Data
pixels=digits.iloc[:,2:]
import numpy as np
pixels = np.asarray(pixels, 'float32')
X = pixels/255.0
label=digits['label'].tolist()
X_train, X_test, Y_train, Y_test = train_test_split(X, label,test_size=0.2,random_state=0)

# Models we will use
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# Hyper-parameters set by cross-validation:
#RBM
learning = np.logspace(-3, 0, 13)
n=range(1, 50, 4)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(rbm_learning_rate=learning, rbm_n_iter=n)
cv = StratifiedShuffleSplit(label, n_iter=5, test_size=0.2, random_state=42)
from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(classifier,param_grid =param_grid, cv=3, scoring='accuracy')
grid.fit(X, label)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

learning = np.logspace(-3, 0, 13)
n=range(1, 50, 4)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(rbm__learning_rate=np.logspace(-3, 0, 13), rbm__n_iter=range(1, 50, 4))
cv = StratifiedShuffleSplit(label, n_iter=5, test_size=0.2, random_state=42)
from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(classifier,param_grid =param_grid, cv=3, scoring='accuracy')
grid.fit(X, label)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
c_range = np.logspace(-3, 1000, 100)
c_scores = []
for c in c_range:
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', linear_model.LogisticRegression(C=c))])
    scores = cross_val_score(classifier,pixels, label, cv=10, scoring='accuracy')
    c_scores.append(scores.mean())
print c_scores

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(c_range, c_scores)
plt.xlabel('Value of C for LogReg')
plt.ylabel('Cross-Validated Accuracy')