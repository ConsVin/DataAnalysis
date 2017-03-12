# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:08:03 2017

@author: Const
"""
#%% Import features list
import pandas
df = pandas.read_csv('features.csv', index_col='match_id')
X = df.copy();
skipFieldList = ['tower_status_radiant',
                 'tower_status_dire',
                 'barracks_status_radiant',
                 'barracks_status_dire',
                 'duration',
                 'radiant_win'];
for f in skipFieldList:
    X.drop(f,1,inplace=True)
X = X.fillna(0);
TargetField = 'radiant_win';
y = df[TargetField];
#%% Get list of non-fullfill field
fList = df.count();
[Len, NF] = df.shape;
print 'List of fields, which not filled for all matches'
notFullFilled = fList[(fList!=Len)];
print notFullFilled
#%% Try Gradien Boosting
import datetime
start_time = datetime.datetime.now()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
n_folds = 5;
kf = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=1);
scoreList = list()
timeList =list();
nEstimList =[ 5,10,20,30,50,100,150]
for n in  nEstimList  :
    clf = GradientBoostingClassifier(n_estimators=n, random_state=0);
    score = 0;
    start_time = datetime.datetime.now();
    print 'Start gredient boosting with n=',n
    for train_index, test_index in kf:
        X_train, X_test = X.values[train_index,:], X.values[test_index,:]
        y_train, y_test = y.values[train_index], y.values[test_index]

        clf.fit(X_train,y_train)
        y_pred = clf.predict_proba(X_test)[:,1];
        s = roc_auc_score(y_test, y_pred)
        score = score + s;
    t = (datetime.datetime.now() - start_time).seconds;
    timeList.append(t);
    print 'Time elapsed:', t,' sec.'
    scoreList.append(score/n_folds)
# Plot Results
import matplotlib.pyplot as plt
plt.plot(nEstimList,scoreList,'-ro')
plt.title('Random Forest Classifier')
plt.ylabel ('AUC ROC')
plt.xlabel ('n_estimatorss')
plt.show()
#%% Part 2.
# 1. Let's try logistic regression

def LogRegressionCrossValiedation(X,y,Clist,nFolds):
	from sklearn.metrics import roc_auc_score
	from sklearn import linear_model
	ScoreList = list();
	kf = KFold(len(X), n_folds=n_folds, shuffle=True);
	for C in Clist:
		logreg = linear_model.LogisticRegression(penalty='l2',C=C)
		score = 0;
		start_time = datetime.datetime.now();
		print 'Start logistic regression with C=',C
		for train_index, test_index in kf:
			X_train, X_test = X[train_index,:], X[test_index,:]
			y_train, y_test = y[train_index]  , y[test_index]

			logreg.fit(X_train, y_train)

			y_pred = logreg.predict_proba(X_test)[:,1];
			
			s = roc_auc_score(y_test,y_pred);
			score = score + s;
    
		t = (datetime.datetime.now() - start_time).seconds;
		sAvg = score/nFolds;
		print'Average Score = ',sAvg
		print 'Time elapsed:', t 
		ScoreList.append(sAvg)
	return ScoreList

nFolds = 5;
CCoefList = [1E-7,1E-5,1E-3,1E-2,1,10,20];
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
Xscaled = scaler.transform(X);
LogScoreList = LogRegressionCrossValiedation(Xscaled,y.values,CCoefList,nFolds);


import matplotlib.pyplot as plt
plt.plot(LogScoreList,'-ro')
plt.title('Logistic Regression Classifier')
plt.ylabel ('AUC ROC')
plt.xlabel ('C values')
plt.show()
#%% Part 2.2.
#Remove categorial values
def drop_categorial(X):
    Xdroped = X.copy();
    skipFieldList = ['lobby_type',
                     'r1_hero',
                     'r2_hero',
                     'r3_hero',
                     'r4_hero',
                     'r5_hero',
                     'd1_hero',
                     'd2_hero',
                     'd3_hero',
                     'd4_hero',
                     'd5_hero'];
    for f in skipFieldList:
        Xdroped.drop(f,1,inplace=True)
    return Xdroped
Xdroped = drop_categorial(X);
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xdroped)
XDropedScaled = scaler.transform(Xdroped);
nFolds = 5;
CCoefList = [1E-7,1E-5,1E-3,1E-2,1,10,20];
LogScoreList = LogRegressionCrossValiedation(XDropedScaled,y.values,CCoefList,nFolds);
import matplotlib.pyplot as plt
plt.plot(LogScoreList,'-ro')
plt.title('Logistic Regression Classifier with droped categorial values')
plt.ylabel ('AUC ROC')
plt.xlabel ('C values')
plt.show()
#%%
# N — количество различных героев в выборке
import numpy as np
HerosList = ['r1_hero',
             'r2_hero',
             'r3_hero',
             'r4_hero',
             'r5_hero',
             'd1_hero',
             'd2_hero',
             'd3_hero',
             'd4_hero',
             'd5_hero'];
Nunique= (np.unique(X[HerosList].values)).size;
Nmax   = np.max(X[HerosList].values);
matrix = X[HerosList].values ;
print 'Unique heroes in set', N
print 'Maximum hero number is ',Nmax
#%%
def get_bagOfHeroes(X,N):
    X_pick = np.zeros((X.shape[0], Nmax))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return X_pick

X_pick = get_bagOfHeroes(X,Nmax);
#%%
bagOfWordsX = np.concatenate( (XDropedScaled,X_pick) , axis = 1);
CCoefList = [1E-7,1E-5,1E-2,1E-3,1E-2,1];
nFolds = 5;
LogScoreList = LogRegressionCrossValiedation(bagOfWordsX,y.values,CCoefList,nFolds);
import matplotlib.pyplot as plt
plt.plot(LogScoreList,'-ro')
plt.title('Logistic Regression Classifier with bag of words')
plt.ylabel ('AUC ROC')
plt.xlabel ('C values')
plt.show()

#%% Get best model
from sklearn import linear_model                            
logreg = linear_model.LogisticRegression(penalty='l2',C=0.01)
logreg.fit(bagOfWordsX,y.values)
y_pred = logreg.predict_proba(bagOfWordsX)[:,1];
s      = roc_auc_score(y.values,y_pred)
print s
#%% Prepare test data

X_test = pandas.read_csv('features_test.csv',index_col='match_id')
X_test = X_test.fillna(0);
X_test_droped = drop_categorial(X_test);
scaler = preprocessing.StandardScaler().fit(Xdroped)
X_test_scaled = scaler.transform(X_test_droped);
bagOfTestHeros = get_bagOfHeroes(X_test,Nmax);
X_test_full = np.concatenate((X_test_scaled,bagOfTestHeros) , axis = 1);             
y_pred_test = logreg.predict_proba(X_test_full)[:,1];
print 'Minimum value of win propability = ', np.min(y_pred_test)
print 'Maximum value of win propability = ', np.max(y_pred_test)

plt.hist(y_pred,bins=100);
plt.title('prediction value distribution')
plt.ylabel ('Number of matches in bins')
plt.xlabel ('1% bin')
plt.show()
#%%
clf = GradientBoostingClassifier(n_estimators=600,max_features = 50);
clf.fit(bagOfWordsX, y.values)
y_pred = clf.predict_proba(bagOfWordsX)[:,1];
#%%
y_pred_test = clf.predict_proba(X_test_full)[:,1];                          
s = roc_auc_score(y.values,y_pred);
print s
#%%
import pandas as pd
y_pred_test = clf.predict_proba(X_test_full)[:,1];
colNames = list();
colNames.append('radiant_win');
dfRes = pd.DataFrame(y_pred_test, index = X_test.index,columns=colNames);
dfRes.to_csv('test_results2.csv')
