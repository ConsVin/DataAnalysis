# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:41:02 2017

@author: Const
"""

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score as score
from sklearn import preprocessing as prp
from sklearn.preprocessing import StandardScaler

Data_train = np.loadtxt('perceptron-train.csv', delimiter=",")
Data_test  = np.loadtxt('perceptron-test.csv' , delimiter=",")


X_train = Data_train[:,1:];
y_train = Data_train[:,0 ];
X_test  = Data_test [:,1:];
y_test  = Data_test [:,0 ];

#clf = Perceptron(random_state  = 241)

clf = Perceptron()
clf.fit(X_train,y_train)
y_predict_1 = clf.predict(X_test)
s1 = score(y_predict_1,y_test)

# Now calc normalized values
#X_train_scaled = prp.scale(X_train);
#X_test_scaled  = prp.scale(X_test);
scaler = StandardScaler();
X_train_scaled  = scaler.fit_transform( X_train )
X_test_scaled   = scaler.fit_transform( X_test  )
clf.fit(X_train_scaled,y_train)
y_predict_2 = clf.predict(X_test_scaled)
s2 = score(y_predict_2,y_test)

delta = s2-s1;

print 'Raw X p=%2.3f | Normalized X p = %2.3f | delta = %2.3f'%(s1,s2,delta)

f = open('W2_3.txt','w+')
f.write('%2.3f' % (delta))
f.close()

print 'Done Ok'
