# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:17:54 2017

@author: Const
""" 
#%% Read data
import pandas as pd
data_train = pd.read_csv('salary-train.csv');

#%% Replace with one-hot categorial (Place, Time)
from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer();

#%% Keep only symbols in the list
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]',' ',regex=True);
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'));
#%% Vectorize text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 5);
TrainDescription = vectorizer.fit_transform(data_train.FullDescription);

#%% Combine all together
from scipy.sparse import hstack
Xfull = hstack((X_train_categ,TrainDescription));
y =data_train.SalaryNormalized;
#%%
# http://scikit-learn.org/stable/modules/linear_model.html
from sklearn import linear_model
clf = linear_model.Ridge(alpha=1);
clf.fit(Xfull,y);
#%%
data_test = pd.read_csv('salary-test-mini.csv');
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]',' ',regex=True);
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'));
TestDescription = vectorizer.transform(data_test.FullDescription);
XFullTest = hstack((X_test_categ,TestDescription));
yTest = clf.predict(XFullTest);
#%%
f = open('sallary.txt','w+')
f.write('%2.2f %2.2f' %  (yTest[0],yTest[1]))
f.close()
