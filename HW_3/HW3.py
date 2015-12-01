# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:33:33 2015

@author: thyme

data.info()
data.describe
data.iloc[:,337].describe()
len(data.columns)
len(data.index)
for i in range(len(data.columns)):
    if sum(data.iloc[:,i].isnull())>0:
        print data.iloc[:,i].describe()

data=data.iloc[:,0:338]
"""

import pandas as pd
data=pd.read_csv('/home/thyme/Stat Analytics/Strong.csv')
data=data.iloc[:,0:338]

filter = data["FelRecidYr3"] != " "
data = data[filter]

data=data.replace(' ','NaN')
felrec=data['FelRecidYr3']
X=data.drop('FelRecidYr3', axis=1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, felrec, random_state=1234567)

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

#Logistic Regression:

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob_log = logreg.predict_proba(X_test)[:, 1]

from sklearn.cross_validation import cross_val_score
logscores = cross_val_score(logreg, X, felrec, cv=5, scoring='accuracy')
logscores
logscores.mean()

#k nearest neighbors:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
# search for an optimal value of K for KNN
k_range = range(1, 50)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, felrec, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# 10-fold cross-validation with the best KNN model
#knn = KNeighborsClassifier(n_neighbors=?)
knnscores = cross_val_score(knn, X, felrec, cv=10, scoring='accuracy')
knnscores
knnscores.mean()

#SVC Gaussian
import numpy as np
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(felrec, n_iter=5, test_size=0.2, random_state=42)
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, felrec)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
from sklearn import svm
clf = svm.SVC(C=?, gamma=?, kernel='rbf')
from sklearn.cross_validation import cross_val_score
svcgscores = cross_val_score(clf, X, felrec, cv=10, scoring='accuracy')
svcgscores
svcgscores.mean()

#SVC Polynomial
import numpy as np
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-3, 2, 13)
from sklearn.cross_validation import StratifiedShuffleSplit
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(felrec, n_iter=5, test_size=0.2, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
grid = GridSearchCV(SVC(kernel='poly'),param_grid=param_grid, cv=cv)
grid.fit(X, felrec)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

from sklearn import svm
clf = svm.SVC(C=?, degree=?, kernel='poly')
from sklearn.cross_validation import cross_val_score
svcgscores = cross_val_score(clf, X, felrec, cv=10, scoring='accuracy')
svcgscores
svcgscores.mean()

# Decision Trees:

from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)


from sklearn.cross_validation import cross_val_score
treescores = cross_val_score(treeclf, X, felrec, cv=10, scoring='accuracy')
treescores
treescores.mean()

#Bagged Decision Tree

from sklearn.ensemble import BaggingClassifier

#Find best n_estimators
potential_n = range(10, 210, 10)
Accuracy_scores = []
for n in potential_n:
    bagclass = BaggingClassifier(treeclf,n_estimators=n, random_state=1)
    scores = cross_val_score(bagclass, X, felrec, cv=2, scoring='accuracy')
    Accuracy_scores.append(scores.mean())

# plot n_estimators (x-axis) versus Accuracy (y-axis)
import matplotlib.pyplot as plt
plt.plot(potential_n, Accuracy_scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')

bagging=BaggingClassifier(treeclf,n_estimators=?, random_state=1)
from sklearn.cross_validation import cross_val_score
bagscores = cross_val_score(bagging, X, felrec, cv=10, scoring='accuracy')
bagscores
bagscores.mean()

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()

#Find best n_estimators
potential_n = range(10, 210, 10)
Accuracy_scores = []
for n in potential_n:
    rfclass = RandomForestClassifier(n_estimators=n, random_state=1)
    scores = cross_val_score(rfclass, X, felrec, cv=2, scoring='accuracy')
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
    scores = cross_val_score(rfclass, X, felrec, cv=10, scoring='accuracy')
    Accuracy_scores.append(scores.mean())

# plot max_features (x-axis) versus Accuracy (y-axis)
plt.plot(feature_range, Accuracy_scores)
plt.xlabel('max_features')
plt.ylabel('accuracy')

rfclass=RandomForestCLassifier(n_estimators=?, max_features=?, random_state=1)
from sklearn.cross_validation import cross_val_score
rfscores = cross_val_score(rfclass, X, felrec, cv=10, scoring='accuracy')
rfscores
rfscores.mean()