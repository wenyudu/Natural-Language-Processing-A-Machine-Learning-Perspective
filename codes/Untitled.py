#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# KL-divergence using numpy and scipy</font>
 
import numpy as np
import scipy.stats

x = [np.random.randint(1, 11) for i in range(10)]
print(x)
# [1, 10, 9, 6, 4, 1, 5, 7, 3, 1]

print(np.sum(x))
# 47

px = x / np.sum(x)

print(px)
# [0.0212766  0.21276596 0.19148936 0.12765957 0.08510638
 # [0.0212766 0.10638298 0.14893617 0.06382979 0.0212766 ]
 

 
y = [np.random.randint(1, 11) for i in range(10)]

print(y)
# [5, 3, 5, 10, 1, 3, 4, 10, 8, 9]

print(np.sum(y))
# 58

py = y / np.sum(y)

print(py)
# [0.0862069  0.05172414 0.0862069  0.17241379 0.01724138
# 0.05172414 0.06896552 0.17241379 0.13793103 0.15517241]

KL = scipy.stats.entropy(px, py)
KL = 0.0
for i in range(10):
    KL += px[i] * np.log(px[i]/py[i])
    
print(KL)
# 0.4354286501297302

