#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Regularization in pytorch</font>

 
loss_fn = nn.CrossEntropyLoss()
reg_loss = 0
for param in model.parameters():
    reg_loss += loss_fn(param)
lambda = 5e-4
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss = loss + lambda*reg_loss
    
    loss.backward()
    optimizer.step()
 
# `lambda` is the regularization parameter that reduces overfitting

 

# Aspects we should consider:
# * different training objectives (large margin or log-likelihood)
# * different feature definitions
# * different hyperparameters (number of training iterations, learning rate)


# **Significance test**
# *Empirical errors* and *generalization errors*
# *pairwise t-test* 
# using the significance level to measure the degree of generalizability.

 
import the packages
 
import numpy as np
from scipy import stats
 
# Define 2 random distributions with size N
```
N = 10
a = np.random.randn(N) + 2
b = np.random.randn(N)
 
print(a,b)

[1.36929026 0.45171207 1.980284   2.69894659 3.58459313 
1.02503278 1.5443674  3.75178741 0.79744317 1.48595977]

[ 1.97056194 -0.44926516 -0.1049745  -1.02613361 -2.1329443
0.7991724 0.60720041  1.22176851 -1.22300267 -0.21052362]
 
# Calculate the standard derivation
 
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)
s = np.sqrt((var_a + var_b)/2)
 

 
# Calculate the t-statistics and compare with the critical t-value
 
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))

df = 2*N -2
p = 1 - stats.t.cdf(t, df=df)
 
# Result:
 
print("t = " + str(t))
print("p = " + str(2*p))

t = 4.26411975457851
p = 0.0004668051407590301
 

 

# Cross checking with the internal scipy function

 
t2, p2 = stats.ttest_ind(a, b)
 

# Result:

 
print("t = " + str(t2))
print("p = " + str(p2))

t = 4.264119754578509
p = 0.0004668051407589875
 

