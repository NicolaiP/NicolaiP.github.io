# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:04:53 2016

@author: Nicolai
"""

#%matplotlib inline
from __future__ import division, print_function
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import theano
import theano.tensor as T
import lasagne

def plot_decision_boundary(pred_func, X, y):
    #from https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    yy = yy.astype('float32')
    xx = xx.astype('float32')
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

# %% 
# Generate a dataset and plot it
np.random.seed(0)
num_samples = 300

X, y = sklearn.datasets.make_moons(num_samples, noise=0.20)

X_tr = X[:100].astype('float32')
X_val = X[100:200].astype('float32')
X_te = X[200:].astype('float32')

y_tr = y[:100].astype('int32')
y_val = y[100:200].astype('int32')
y_te = y[200:].astype('int32')

plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.BuGn)

print("Shape of the input matrix dim(%i x %i)." % X.shape)
print("Shape of the target matrix dim(%i)." % y.shape)

num_features = X_tr.shape[-1]


# %%

from lasagne.updates import sgd # stochastic gradient descent optimization algorithm.
from lasagne.nonlinearities import leaky_rectify, softmax, tanh
from lasagne.layers import InputLayer, DenseLayer


#MODEL SPECIFICATION
l_in = InputLayer(shape=(None, num_features))

# Insert hidden layer with five hidden units
l_hid = DenseLayer(incoming=l_in, num_units=10, nonlinearity=tanh, name='hiddenlayer') 

#l = DenseLayer(incoming=l,.....
l_out = DenseLayer(incoming=l_hid, num_units=2, nonlinearity=softmax, name='outputlayer') 
#We use two output units since we have two classes. The softmax function ensures that the the class probabilities sum to 1.
    