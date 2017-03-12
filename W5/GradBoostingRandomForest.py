# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 10:11:35 2017

@author: Const
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigma_func(x):
    return float(1) / (float(1) + np.exp(-x))
    
PDdata = pd.read_csv('gbm-data.csv');
data = PDdata.as_matrix();

X = data[:,1:]
y = data[:,0]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size = 0.8, random_state=241);


from sklearn.ensemble import GradientBoostingClassifier    
from sklearn.metrics import log_loss
LearningRates = [ 1, 0.5,  0.3, 0.2, 0.1 ];
nEstim = 250;
nLR    = len(LearningRates);

test_loss_score     = np.empty([nLR,nEstim])
test_logloss_score  = np.empty([nLR,nEstim])
train_loss_score    = np.empty([nLR,nEstim])
train_logloss_score = np.empty([nLR,nEstim])

common_args = {'n_estimators': nEstim, 'random_state': 241,'verbose': True }
for i in range(nLR):
    LRcurr = LearningRates[i];
    clf = GradientBoostingClassifier(learning_rate=LRcurr, **common_args);
    clf.fit(X_train,y_train);
    for j, pred in enumerate(clf.staged_decision_function(X_test)):
        test_loss_score[i,j] = clf.loss_(y_test, pred)
        test_logloss_score[i,j] = log_loss(y_test, sigma_func(pred));
    for j, pred in enumerate(clf.staged_decision_function(X_train)):
        train_loss_score[i,j]    = clf.loss_(y_train, pred)
        train_logloss_score[i,j] = log_loss(y_train, sigma_func(pred));

plt.figure()
plt.plot(test_loss_score[3,:].T, linewidth=2)

idx = 3;
minLogLossTest = min(test_logloss_score[idx,:])
nIterMin       =  np.where(test_logloss_score[idx,:] == minLogLossTest)
print 'At Learning Rate %2.2f minimum logloss %2.2f at iter = %d'%(LearningRates[idx],minLogLossTest,nIterMin[0])

#%% Learn Random Forest Classifier
nTrees = nIterMin[0];
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = nTrees, random_state=241)
clf.fit(X_train,y_train);
RandomForestPred = clf.predict_proba(X_test)
ResRandForest = log_loss(y_test,RandomForestPred[:,1]);
print 'Random forest Result %2.2f' % (ResRandForest)