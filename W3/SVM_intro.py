# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:38:22 2017

@author: Const
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = np.loadtxt('svm-data.csv', delimiter=",")
y = data[:,0]
X = data[:,1:]
clf = SVC(C = 100000,random_state=241)
clf.fit(X, y)  

f = open('W3_1.txt','w+')
for item in clf.support_:
  f.write("%d " % (item+1))
f.close()