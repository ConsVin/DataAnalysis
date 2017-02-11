# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:33:27 2017

@author: Const
"""
#%% Intro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as aucroc
from sklearn.metrics import precision_recall_curve as prCurve

#%% Part 1, Load Values
data = np.loadtxt('classification.csv', delimiter=",",skiprows = 1)
y_true = data[:,0];
y_pred = data[:,1];

#%%Part 2, Confusion Matrix

CMatrix = confusion_matrix (y_true, y_pred);
TN = CMatrix[0,0];
FP = CMatrix[0,1];
TP = CMatrix[1,1];
FN = CMatrix[1,0];
f = open('res_MET_1.txt','w+')
f.write("%d %d %d %d" % (TP, FP, FN, TN))
f.close()

#%%Part 3, Main Metrics
Accuracy  = accuracy_score ( y_true, y_pred);
Precision = precision_score( y_true, y_pred);
Recall    = recall_score   ( y_true, y_pred);
F1_score  = f1_score       ( y_true, y_pred);
f = open('res_MET_2.txt','w+')
f.write("%2.2f %2.2f %2.2f %2.2f" % (Accuracy, Precision, Recall, F1_score));                        
f.close()

#%%Part 4
#scores = np.loadtxt('scores.csv', delimiter=",",skiprows = 1)
df = pd.read_csv('scores.csv');
y_true = df.true;
#%% Part 5
AucRocList = list()
for column in df:
    a = aucroc(y_true,df[column]);
    AucRocList.append(a);
AucRocList[0] = 0;
MaxInd =  AucRocList.index(max(AucRocList))
NameOfBest = df.columns[MaxInd];
f = open('res_MET_3.txt','w+')
f.write("%s" % NameOfBest);                        
f.close()

#%% part 6
# Which classificator has maximum P and R>0.7
df = pd.read_csv('scores.csv');
BestPrecList = list()
y_true = df.true;
for y_scores in df:
    [precision, recall, thresholds] = prCurve(y_true, df[y_scores]);
    BestPrec = max(precision[np.where(recall>0.7)])
    BestPrecList.append(BestPrec)
BestPrecList [0]   = 0;
MaxInd = BestPrecList.index(max(BestPrecList));
NameOfBest = df.columns[MaxInd];
f = open('res_MET_4.txt','w+')
f.write("%s" % NameOfBest);                        
f.close()

