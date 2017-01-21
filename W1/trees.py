# Step 1. 
import pandas as pd
import numpy as np
dataRaw = pd.read_csv('titanic.csv',index_col='PassengerId');
# Step 2.
data = dataRaw[['Survived','Pclass','Fare','Age','Sex']]
data.Sex = data.Sex.replace('female',1);
data.Sex = data.Sex.replace('male',0);
data = data.dropna(axis=0);


# Get desired pair of signals
X = data[[1,2,3,4]].values;
y = data[[0]].values;
from sklearn.tree import DecisionTreeClassifier
 # Train
clf =DecisionTreeClassifier(random_state=241, splitter='best');
clf.fit(X, y)
print clf.feature_importances_
#------------
