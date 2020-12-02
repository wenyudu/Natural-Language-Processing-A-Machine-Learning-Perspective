#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The solution to the coin tossing problem using EM algorithm in python.
# 1. import the packages we need

 
import numpy as np
import math
import matplotlib.pyplot as plt
 

# 2. define the binomial log likelihood which return the likelihood of obserations given the probabilities.
 
def get_binomial_log_likelihood(obs, probs):
    N = sum(obs)
    k = obs[0]
    binomial_coeff = math.factorial(N) /
    (math.factorial(N-k)*math.factorial(k))
    prod_probs = obs[0]*math.log(probs[0]) + 
    obs[1]*math.log(1-probs[0])
    log_lik = binomial_coeff + prod_probs
    
    return log_lik
 

 

# 3. simulate experiment data:
# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
# 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
# 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
# 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
# 5th:  Coin A, {THHHTHHHTH}, 7H,3T

 
head_counts = np.array([5,9,8,4,7])
tail_counts = 10-head_counts
experiments = list(zip(head_counts, tail_counts))

pA_heads = np.zeros(100)
pA_heads[0] = 0.60
pB_heads = np.zeros(100)
pB_heads[0] = 0.50
```
from MLE: pA(heads) = 0.80 and pB(heads)=0.45


 

# 4. define some hyper parameters
 
delta = 0.001
j = 0

improvement = float('inf')
 
# `delta` : the lower bound of improvement that should be accepted
# `j` : iteration counter
 

 
# 5. implement the EM algorithm
 
while (improvement>delta):
    expectation_A = np.zeros((len(experiments),2),
                    dtype=float)
    expectation_B = np.zeros((len(experiments),2),
                    dtype=float)
    for i in range(0,len(experiments)):
        e = experiments[i]
        ll_A = get_binomial_log_likelihood(e,
               np.array([pA_heads[j], 1-pA_heads[j]]))
        ll_B = get_binomial_log_likelihood(e,
               np.array([pA_heads[j], 1-pA_heads[j]]))
        weightA = math.exp(ll_A) / 
                  (math.exp(ll_A) + math.exp(ll_B))
        weightB = math.exp(ll_B) /
                  (math.exp(ll_A) + math.exp(ll_B))
        
        expectation_A[i] = np.dot(weightA, e)
        expectation_B[i] = np.dot(weightB, e)
 
 
    pA_heads[j+1] = sum(expectation_A)[0] / 
                    sum(sum(expectation_A))
    pB_heads[j+1] = sum(expectation_B)[0] / 
                    sum(sum(expectation_B))
    
    improvement = (max(abs(np.array([pA_heads[j+1], 
                  pB_heads[j+1]]) -
                  np.array([pA_heads[j],pB_heads[j]]))))
    
    j = j + 1
 

# 6. plot the result
 
plt.figure()
plt.plot(range(0,j),pA_heads[0:j], 'r--')
plt.plot(range(0,j),pB_heads[0:j])
plt.show()
 

