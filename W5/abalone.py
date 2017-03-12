# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 09:19:20 2017

@author: Const
"""

##  Step 1. Load abalonce
import pandas as pd
data = pd.read_csv('abalone.csv');
#step 2. Sex to digit
data.Sex = data.Sex.map(lambda x:  1 if x == 'M' else (-1 if x == 'F' else 0));
# 3. Divide to X,Y
y = data.Rings;
X = data.iloc[:,0:-1]

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

scoreList = list()
n_folds = 5
kf = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=1);
threshold = 0.52;
minimumValue = 0;          
for n in range(1,50):
    clf = RandomForestRegressor(n_estimators = n, random_state = 1)
    score = 0;
    for train_index, test_index in kf:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score = score + r2_score(y_test,y_pred)
    score = score / n_folds;
    scoreList.append(score)
    print ('n = ',n)
    print ('score = ',score)

import matplotlib.pyplot as plt
plt.plot(scoreList)
plt.title('Random Forest Classificator')
plt.ylabel ('R2 score')
plt.xlabel ('N of enstimators')
plt.show()