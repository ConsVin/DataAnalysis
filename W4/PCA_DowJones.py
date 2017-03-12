# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:09:07 2017

@author: Const
"""
#%% Load data
import pandas as pd
data_train = pd.read_csv('close_prices.csv',parse_dates  = True,index_col = 'date' );

#%% Run PCA and get minumum number of components to achive threshold Variace
from sklearn.decomposition import PCA
nComp = 10;
pca = PCA(n_components = nComp);
pca.fit(data_train)
threshold = 0.9;
print "Explained ratios"
print pca.explained_variance_ratio_
for x in xrange(1,nComp):
    explainedVariance = sum(pca.explained_variance_ratio_[0:x]);
    print "Explained variance with %d components is %2.3f" %(x,explainedVariance)
    if (explainedVariance > threshold):
        nEnoughComp = x;
        print "That's enough!"
        break
f = open('DowJones_ncomp.txt','w+')
f.write('%d' %  nEnoughComp)
f.close()

#%% Apply fited PCA to data
data_PrincipalComponents = pca.transform(data_train);

#%% Get Pearson correlation
from scipy.stats import pearsonr
DowJonesIndex = pd.read_csv('djia_index.csv',parse_dates  = True,index_col = 'date' );
corr = pearsonr(data_PrincipalComponents[:,0], DowJonesIndex.DJI);
f = open('DowJones_corr.txt','w+')
f.write('%2.2f' %  corr[0])
f.close()

#%% Find company
import numpy as np
comp = pca.components_ ;
n = 0;
nMax = np.argmax(comp[n,:]);
CompanyIndex = data_train.columns.values[nMax];
CompanyName = 'Visa';
print '%s means %s' %(CompanyIndex,CompanyName)
f = open('DowJones_name.txt','w+')
f.write('%s' %  CompanyName)
f.close()

