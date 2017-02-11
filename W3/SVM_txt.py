# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:04:40 2017

@author: Const
"""
#%%
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.grid_search import GridSearchCV
#%%
newsgroups = datasets.fetch_20newsgroups( subset = 'all',
                                         categories = ['alt.atheism','sci.space'])
#%% 
vectorizer = TfidfVectorizer();
X = vectorizer.fit_transform(newsgroups.data);
y = newsgroups.target;
#%%
grid = {'C': np.power(10.0, np.arange(-5, 6))};
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241);
clf = svm.SVC(kernel='linear', random_state=241);
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv);
clf.fit(X, y);
a = abs(clf.coef_.toarray());
indices= np.argsort(a);
#%%
features = vectorizer.get_feature_names()
fList = list();
top_n = 10
for i in indices[0,-top_n:]:
    fList.append(features[i]);
BestWords = sorted(fList)
#%%
f = open('res_TXT.txt','w+')
for s in BestWords:
    f.write("%s " % s);                        
f.close()
