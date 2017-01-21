# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:50:50 2017

@author: Const
"""
WEEK = 2;
PART = 2;

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor  as KNR
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

boston = load_boston();
X = scale(boston.data);
y = scale(boston.target);

pRange = np.linspace(1,10,200);

kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42);
kMeans = list()
for p in pRange:
    knr = KNR(n_neighbors = 5, weights = 'distance',metric  = 'minkowski', p = p);
    knr.fit(X,y) # Train classifier
    arr = cross_val_score(estimator = knr, X=X,y=y,scoring='mean_squared_error', cv=kf)#m = arr.mean();
    m = arr.mean();
    kMeans.append(m)
    

plt.plot(pRange,kMeans)
plt.xlabel('p in Minkowski metric')
plt.ylabel('Quality')

maxValue = max(kMeans);
optP     = pRange[kMeans.index(maxValue)];
print "Maximum value is %2.2f at p=%d" % (maxValue,optP)

f = open('W2_2_1.txt','w+')
f.write('%d' % (optP))
f.close()

print 'Done'