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
import matplotlib.pyplot as plt

Data_train = np.loadtxt('perceptron-train.csv', delimiter=",")
Data_test  = np.loadtxt('perceptron-test.csv' , delimiter=",")

plot_colors = 'br';
classes = [-1,1];
X_train = Data_train[:,1:];
y_train = Data_train[:,0 ];
X_test  = Data_test [:,1:];
y_test  = Data_test [:,0 ];


clf = Perceptron(random_state  = 241)
#clf = Perceptron()

clf.fit(X_train,y_train)
yRawPredTrain = clf.predict(X_train)
yRawPredTest  = clf.predict(X_test)
s0 = score(yRawPredTrain,y_train)
s1 = score(yRawPredTest,y_test)

print 'Raw   X, train p=%2.3f ' % s0 
print 'Raw   X, test  p=%2.3f ' % s1 

# Now calc normalized values
#X_train_scaled = prp.scale(X_train);
#X_test_scaled  = prp.scale(X_test);
scaler = StandardScaler();
scaler.fit_transform(X_test )
print 'Test Data scaler mean and variance'
print scaler.mean_
print scaler.std_

X_train_scaled  = scaler.fit_transform( X_train )
X_test_scaled   = scaler.transform( X_test  )
print 'Train Data scaler mean and variance'
print scaler.mean_
print scaler.std_



clf.fit(X_train_scaled,y_train)
yTrainPredict = clf.predict(X_train_scaled)
yTestPredict  = clf.predict(X_test_scaled)
s2 = score(yTrainPredict,y_train)
s3 = score(yTestPredict,y_test)

d0 = s2-s0;
d1 = s3-s1;
print 'Scale X, train p=%2.3f ' % s2
print 'Scale X, test  p=%2.3f ' % s3 
print 'Train delta    d=%2.3f ' % d0
print 'Test  delta    d=%2.3f ' % d1


f = open('W2_3.txt','w+')
f.write('%2.3f' % (d1))
f.close()

## Plot the surface
plot_step = 0.01;
x_min, x_max = X_train_scaled[:,0].min()-1,X_train_scaled[:,0].max()+1
y_min, y_max = X_train_scaled[:,1].min()-1,X_train_scaled[:,1].max()+1
xx,yy = np.meshgrid(np.arange (x_min, x_max, plot_step),
                    np.arange (y_min, y_max, plot_step));
x_plot = np.c_[xx.ravel(),yy.ravel()];
Z = clf.predict(x_plot)
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
X = X_train_scaled;
y = y_train;
for i,color in zip(classes,plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx,0],X[idx,1],c=color)
plt.title('Train data')
plt.show()
##
cs_test = plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
X = X_test_scaled;
y = y_test;
for i,color in zip(classes,plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx,0],X[idx,1],c=color)
plt.title('Test data')
plt.show()
##

seedMax = 20000;
scoreList = list()
for seed in range(1,seedMax):
    clf = Perceptron(random_state  = seed)
    clf.fit(X_train_scaled,y_train)
    yTestPredict  = clf.predict(X_test_scaled)
    rez = score(yTestPredict,y_test)
    scoreList.append(rez);
a = np.asarray(sorted(scoreList));
plt.plot(a)
plt.grid()
plt.show()
nMax=sum(a==max(a));

aUniq = np.unique(a*len(y_test))
print 'Best Result %2.6f at %d seeds , chance %2.2f%%' % (max(a),nMax,float(nMax)/seedMax*100)


## Plot 
print 'Done Ok'
