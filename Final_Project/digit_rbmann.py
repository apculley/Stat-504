# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:01:43 2015

@author: thyme
"""

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
c_range = [.0001,.001,.01, .1,1, 10, 100, 1000, 5000, 10000]
c_scores = []
for c in c_range:
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', linear_model.LogisticRegression(C=c))])
    scores = cross_val_score(classifier,pixels, label, cv=2, scoring='accuracy')
    c_scores.append(scores.mean())
print c_scores

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(c_range, c_scores)
plt.xlabel('Value of C for LogReg')
plt.ylabel('Cross-Validated Accuracy')


from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

cross_val_score(classifier,pixels, label, cv=2, scoring='accuracy')