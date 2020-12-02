#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cross entropy implementation with python codes 
import numpy as np
def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions. epsilon, 1.-epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce
 

# `numpy.clip`: clip or limit the values in an array, given an interval, values outside the interval are clipped to the interval edges.

 
# Apply to the test data
 
predictions = np.array([[0.25,0.25.0.25,0.25],
                       [0.01,0.01,0.01,0.97]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
ans = 0.71355817782
x = cross_entropy(predictions, targets)
print(np.isclose(x,ans))

