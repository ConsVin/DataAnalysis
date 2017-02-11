# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 08:58:54 2017

@author: Const
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math

data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]

setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
sns.despine()