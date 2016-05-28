#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pickle 

dataset='notMNIST.pickle'
# Load the notMNIST dataset
with open(dataset, 'rb') as f:
        try:
            data_set= pickle.load(f, encoding='latin1')
        except:
            data_set=pickle.load(f)


# # Use only one feature
diabetes_X_train = data_set.get("train_dataset")
diabetes_X_test = data_set.get("test_dataset")
# reshape the input data
print "Befor",diabetes_X_train.shape

input_shape=28*28
diabetes_X_train=diabetes_X_train.reshape((len(diabetes_X_train), input_shape))
diabetes_X_test=diabetes_X_test.reshape((len(diabetes_X_test), input_shape))

print "After",diabetes_X_train.shape

# # # Split the targets into training/testing sets
diabetes_y_train = data_set.get('train_labels')
diabetes_y_test = data_set.get('test_labels')



# # Create linear regression object
regr = linear_model.LogisticRegression()

# Train the model using the training sets
print type(diabetes_X_train)
print type(diabetes_y_train)
# regr.fit(diabetes_X_train, diabetes_y_train)	

regr.fit(diabetes_X_train[:1000],diabetes_y_train[:1000])
# #The coefficients
print('Coefficients: \n', regr.coef_)
# # The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test[50:115]) - diabetes_y_test[50:115]) ** 2))
# Explained variance score: 1 is perfect prediction$rint('Variance score: %.2f' % regr.score(diabetes_X_test[50:115], diabetes_y_test[50:115]))

# # Plot outputs
# # plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# # plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
# #          linewidth=3)

# # plt.xticks(())
# # plt.yticks(())

# # plt.show()
