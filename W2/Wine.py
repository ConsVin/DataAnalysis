# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:20:19 2017

@author: Const
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
##
data = np.loadtxt('wine.data.txt', delimiter=",")
X = data[:,1:14]
y = data[:,0]

kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42);

# First, without scaling
kMeans = list()
for k in range (1,50):
    knn = KNC( n_neighbors = k ) # Create classifier
    knn.fit(X,y) # Train classifier
    arr = cross_val_score(estimator = knn, X=X,y=y,scoring='accuracy', cv=kf)
    m = arr.mean();
    kMeans.append(m)

plt.plot(kMeans)
plt.xlabel('N of neighbors');
plt.ylabel('Quality');
plt.title('Cross validation quality');
maxValue = max(kMeans);
maxIndex = kMeans.index(maxValue);
optK = maxIndex + 1;
print "Maximum value is %2.2f at %d" % (maxValue,optK)

# Write answers
f = open('W2_1_1.txt','w+')
f.write('%d' % (optK))
f.close()
f = open('W2_1_2.txt','w+')
f.write('%2.2f' % (maxValue))
f.close()

##
X_scaled = scale(X);
kMeans = list()
for k in range (1,50):
    knn = KNC( n_neighbors = k ) # Create classifier
    knn.fit(X,y) # Train classifier
    arr = cross_val_score(estimator = knn, X=X_scaled,y=y,scoring='accuracy', cv=kf)
    m = arr.mean();
    kMeans.append(m)
plt.plot(kMeans)
plt.xlabel('N of neighbors');
plt.ylabel('Quality');
plt.title('Cross validation quality');
maxValue = max(kMeans);
maxIndex = kMeans.index(maxValue);
optK = maxIndex + 1;
print "Maximum value is %2.2f at %d" % (maxValue,optK)

# Write answers
f = open('W2_1_3.txt','w+')
f.write('%d' % (optK))
f.close()
f = open('W2_1_4.txt','w+')
f.write('%2.2f' % (maxValue))
f.close()
#
print 'Done Ok'