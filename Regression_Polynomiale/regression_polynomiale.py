#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:18:34 2021

@author: christiankemgang
"""

# Import lybraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, -1].values

# Building the model
from sklearn .linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) 
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, Y)

# Do a new prediction
#Y_pred = regressor.predict(X_test)
# Predict the new salary after 15 years experience 
#val_pred = [[15]]
#regressor.predict(val_pred)

# Show results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X_poly), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
