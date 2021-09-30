#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Data Preprocessing

"""
Created on Fri Sep 24 21:07:25 2021

@author: christiankemgang
"""

# Import lybraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean', fill_value=None, verbose=0, copy=True)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Handling categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Purchased column
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Divide the dataset between the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





































