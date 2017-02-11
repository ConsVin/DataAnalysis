# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 22:03:17 2017

@author: Const
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as aucroc
from sklearn.metrics import precision_recall_curve as prCurve
#%% 
def sigma_func(x):
    return float(1) / (float(1) + np.exp(-x))

def gradient_descent(X,Y,W,C,k):
    pred = X*W          ; #Scalar Product
    prob = -pred*Y       ; 
    t = float(1) - float(1)/(float(1) + np.exp(prob));
    a    = X*Y; #Scalar XY
    grad = sum(a*t)/Y.size;
    regF = C*W;
    step = k*(grad - regF);
    Wupd = W + step;
    return Wupd
#%% Part 1, Load Values
data = np.loadtxt('data_logistic.csv', delimiter=",",skiprows = 1)
Y = np.zeros((204,1));
Y[:,0]= data[:,0];
y_true = data[:,0];
X = data[:,1:];
#%%
W =np.array([[0,0]]);
dist = list();
threshold = 1E-5;
k = 0.1;
C = 10
for i in xrange(1,10000):
    Wupd = gradient_descent (X,Y,W,C,k);
    d = np.linalg.norm(Wupd-W);
    dist.append(d);
    W = Wupd;
    if (d < threshold):
        break;
print "Stop at %d step" % i        
plt.figure(0)
plt.plot(dist);
yPred = X.dot(W.transpose());
yPredProb = sigma_func(yPred);
plt.figure(1)
plt.plot(yPredProb);
a = aucroc(y_true,yPredProb);
print 'AUC_ROC %2.4f' %a
