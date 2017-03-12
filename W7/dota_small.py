# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:41:24 2017

@author: Const
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:08:03 2017

@author: Const
"""
#%% Import features list
import pandas
features = pandas.read_csv('./features.csv', index_col='match_id')
features.head()
X = features.copy();
skipFieldList = ['tower_status_radiant',
                 'tower_status_dire',
                 'barracks_status_radiant',
                 'barracks_status_dire',
                 'duration',
                 'radiant_win'];
for f in skipFieldList:
    X.drop(f,1,inplace=True)
X = X.fillna(0)

#% Get list of non-fullfill field
fList = X.count();
[Len, NF] = X.shape;
print 'List of fields, which not filled for all matches'
notFullFilled = fList[(fList!=Len)];
print notFullFilled.index
#% Get target
TargetField = 'radiant_win';
y = features[TargetField];
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
nEstimList =[5,7,10,15]
for n in  nEstimList  :
    clf = GradientBoostingClassifier(n_estimators=n, random_state=0);
    score = 0;
    start_time = datetime.datetime.now();
    print ('Start gredient boosting with n=',n)
    for train_index, test_index in kf:
        X_train, X_test = X.values[train_index,:], X.values[test_index,:]
        y_train, y_test = y.values[train_index], y.values[test_index]

        clf.fit(X_train,y_train)
        y_pred = clf.predict_proba(X_test)[:,1];
        s = roc_auc_score(y_test, y_pred)
        score = score + s;
    t = datetime.datetime.now() - start_time;
    timeList.append(t.total_seconds());
    print ('Time elapsed:', t)
    scoreList.append(score/n_folds)
#% Plot Results
import matplotlib.pyplot as plt
plt.plot(nEstimList,scoreList,'-ro')
plt.title('Random Forest Classifier')
plt.ylabel ('AUC ROC')
plt.xlabel ('n_estimatorss')
plt.show()
#%% Part 2.
# 1. Let's try logistic regression
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
LogScoreList = list();
n_folds = 5;
CCoefList = [1E-7,1E-5,1E-3,1E-2,1,10,1E20];
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
Xscaled = scaler.transform(X);
kf = KFold(len(X), n_folds=n_folds, shuffle=True);
for C in CCoefList:

    score = 0;
    start_time = datetime.datetime.now();
    print ('Start logistic regression with C=',C)
    for train_index, test_index in kf:
        X_train, X_test = Xscaled[train_index,:], Xscaled[test_index,:]
        y_train, y_test = y.values[train_index], y.values[test_index]
        logreg = linear_model.LogisticRegression(penalty='l2',C=C)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict_proba(X_test)[:,1];
        s = roc_auc_score(y_test,y_pred);
        print (s)
        score = score + s;
    t = datetime.datetime.now() - start_time;
    timeList.append(t.total_seconds());
    print 'Time elapsed:', t
    LogScoreList.append(score/n_folds)
LogScoreList
plt.plot(LogScoreList,'-ro')
plt.title('Logistic Regression Classifier')
plt.ylabel ('AUC ROC')
plt.xlabel ('C values')
plt.show()
