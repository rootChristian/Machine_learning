#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:45:24 2021

@author: christiankemgang
"""

# Import lybraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Divide the dataset  between the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/3, random_state=0)

# Building the model
from sklearn .linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Do a new prediction
Y_pred = regressor.predict(X_test)
# Predict the new salary after 15 years experience 
#val_pred = [[15]]
#regressor.predict(val_pred)

# Show results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()




































