#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:58:40 2017

@author: a_santos

In this script I try to determinate the best machine learning technique
to achive a good mass ratio estimation from gravitational waves from the
Georgia Tech catalog.

It is necessary to import data preprocessed availables in this repository
*x.npy
*mass_ratio-npy

As result I found that the bes technique is to use a neural network with a
hidden layer with 25 neurons with sigmoidal activation (tanh)

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

#%% Load data
x = np.load('/home/user/.../x.npy')
q = np.load('/home/user/.../mass_ratio.npy')

#%% Data preprocessing
# Concatenate to shuffle the data
data = np.hstack((x.reshape(385, 100), q.reshape(385, 1)))

# Data shuffling
np.random.shuffle(data)

# Spliting data in two subsets, training = 70% of data  and testing = 30% of data    
data_train, data_test = train_test_split(data, test_size=0.3)

# Rearrangement (to present the data to the functions)
x_train = data_train[:, 0:100]
q_train = data_train[:, 100]

x_test = data_test[:, 0:100]
q_test = data_test[:, 100]

#%% Best architecture determination. Each architecture will be tested 10 times
# to assess its performance from 5 to 100 neurons in the hidden layer.

trials = 10
neurons = np.arange(5, 105, 5)
mse = np.zeros([len(neurons), trials])

for k in range(trials):
    j = 0
    for i in neurons:
        MLP = MLPRegressor(hidden_layer_sizes=(i,), activation='tanh', solver='lbfgs')
        MLP.fit(x_train, q_train)
        q_train_out = MLP.predict(x_train)
        mse[j, k] = mean_squared_error(q_train, q_train_out)
        print('Trial = ', k + 1, '   Neurons = ', i)
        j = j + 1
        del(MLP)
            
# Plotting of the MSE of each running
plt.boxplot(mse.T, labels = ['5', '10', '15', 
                             '20', '25', '30', 
                             '35', '40', '45', 
                             '50', '55', '60', 
                             '65', '70', '75', 
                             '80', '85', '90', 
                             '95', '100'])
plt.title('Mean_Squared_Error along Neurons_number')
plt.xlabel('Neurons')
plt.ylabel('MSE')
plt.grid(True)

# The best architecture was the one with 25 neurons

#%% MLP testing with the best architecture
trials = 10
neurons = 5
mse = []

for i in range(trials):
    MLP = MLPRegressor(hidden_layer_sizes=(25,), activation='tanh', solver='lbfgs')
    MLP.fit(x_train, q_train)
    q_test_out = MLP.predict(x_test)
    mse.append(mean_squared_error(q_test, q_test_out))
    print('Trial = ', i + 1)
    del(MLP)

#%%*************************************************************************** 
# kNN. The metric and the neighbors will be varying to determinate the best
# configuration
metrics = np.array([1, 2, 5])
neighbors = np.array([3, 5, 7, 9])


mse = np.zeros([len(neighbors), len(metrics)])
m = 0
for k in metrics:
    l = 0
    for j in neighbors:
        kNN = KNeighborsRegressor(n_neighbors = j, p = k)
        kNN.fit(x_train, q_train)
        q_train_out = np.zeros([len(q_train), 1])
        for i in range(len(q_train)):
            q_train_out[i] = kNN.predict([x_train[i, :]])
        mse[l, m] = mean_squared_error(q_train, q_train_out)
        del(kNN)
        print('metric = ', metrics[m], '  neighbor = ', neighbors[l])
        l = l + 1
    m = m + 1
    
# The best performance was that with p = 5 and neighbors = 3

#%% kNN testing with the best parameters
kNN = KNeighborsRegressor(n_neighbors = 3, p = 5)
kNN.fit(x_train, q_train)
q_test_out = np.zeros([len(q_test), 1])
for i in range(len(q_test)):
    q_test_out[i] = kNN.predict([x_test[i, :]])
mse = mean_squared_error(q_test, q_test_out)
del(kNN)

#%%*************************************************************************** 
# SVM regression. The C parameter will be varying to determinate the best
# configuration

C = np.array([0.1, 0.5, 1., 5., 10., 100.])
mse = []

for i in C:
    SVM_reg = SVR(C=i, kernel='linear')
    SVM_reg.fit(x_train, q_train)
    q_train_out = SVM_reg.predict(x_train)
    mse.append(mean_squared_error(q_train, q_train_out))
    del(SVM_reg)
    print('C = ', i)

# The best performance was that with C = 10

#%% SVM testing with the best configuration
SVM_reg = SVR(C = 10, kernel = 'linear')
SVM_reg.fit(x_train, q_train)
q_test_out = SVM_reg.predict(x_test)
mse = mean_squared_error(q_train, q_train_out)
del(SVM_reg)

#%% Plot the outcome of any algorithm
sort_index_q_test = np.argsort(q_test)
plt.plot(range(len(q_test)), q_test[sort_index_q_test], '.g', range(len(q_test)), q_test_out[sort_index_q_test], '.r')
