# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:42:51 2017

@author: Const
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as aucroc
from sklearn import linear_model

data = np.loadtxt('data_logistic.csv', delimiter=",",skiprows = 1)
Y = data[:,0];
X = data[:,1:];
logreg = linear_model.LogisticRegression(penalty='l2',C = 0.0001)
logreg.fit(X, Y);
yPred= logreg.predict(X);
yPredProb = logreg.predict_proba(X);
s = sum(Y == yPred);
print s
a = aucroc(Y,yPredProb[:,1]);
print 'AUC_ROC %2.3f' %a