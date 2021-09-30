#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:03:32 2021

@author: christiankemgang
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset =  pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Handling categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# State column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

# Divide the dataset between the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Building the model
from sklearn .linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Do a new prediction
Y_pred = regressor.predict(X_test)

# Predict the new salary after 15 years experience 
#val_pred = [[1, 0, 130000,140000, 300000]]
#regressor.predict(val_pred)


