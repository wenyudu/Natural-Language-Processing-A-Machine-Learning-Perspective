#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Gradient descent in python

# Import the packages that we need


# import numpy as np
# import matplotlib.pyplot as plt
 

# Generate data with linear equation $y=\theta_0 + \theta_1X$ and some Gaussian noise, let $\theta_0$ be 4, and $\theta_1=3$

 
X = X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
 

 
# Plot data 
 
plt.plot(X, y, 'b.')
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
 

 
 
# Construct cost function to calculate cost, we use MSE for linear regression. Represent $y$ as $J(\theta)$,  our predition as $h(\theta)$, Here is what the cost function looks like:
# $$
# J(\theta)=1/2m\sum_{i=1}^m(h(\theta)^{(i)}-y^{(i)})^2
# $$

 
def cal_cost(theta, X, y):
  m = len(y)
  
  predictions = X.dot(theta)
  cost = (1/2*m)*np.sum(np.squre(predictions-y))
  return cost
 

# > `theta` is a vector of thetas, we initial it with random thetas
 
# > `m` is the number of observations
 

# Construct stochastic gradient descent function to calculate new Theta vector
 
def sgd(X, y theta, learning_rate=0.01, iterations=10):
    m = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind, :].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            prediction = np.dot(X_i, theta)
            theta = theta - (1/m)*learning_rate*
                    (X_i.T.dot((prediction = y_i)))
            cost += cal_xost(theta, X_i, y_i)
        cost_history[it] = cost
    return theta, cost_history
 

 #> `X`: matrix of X with added bias units
# >
# > `theta`: vector of thetas before learned
 
# > `iterations` : number of iterations

 
# Set the hyper-parameters and see what we can get

 
lr = 0.5
n_iter = 50

theta = np.random.randn(2, 1)

X_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history = sgd(X_b, y, theta, lr, n_iter)

print('Theta0:  {:0.3f},\nTheta1: 
       {:0.3f}'.format(theta[0][0], theta[1][0]))
print('Final cost/MSE: {:0.3f}'.format(cost_history[-1]))
 

# Output:
 
Theta0:  3.986,
Theta1:  3.031
Final cost/MSE: 61.248
 

 
# Mini batch gradient descent

 
def minibatch_gd(X, y, theta, learning_rate=0.01,
                iterations=10, batch_size=20):
    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m/batch_size)
    for it in range(iterations):
        cost = 0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0, m, batch_size):
            X_i = X[i: i+batch_size]
            y_i = y[i: i+batch_size]
            X_i = np.c_[np.ones(len(X_i)), X_i]
            prediction = np.dot(X_i, theta)
            theta = theta - (1/m)*learning_rate*
                    (X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost
    return theta, cost_history
 

# Minibatch SGD with the same hyper-parameters
 
lr = 0.1
n_iter = 200
theta = np.random.randn(2, 1)

theta, cost_history = minibatch_gd(X, y, theta, lr, n_iter)

print('Theta0:  {:0.3f},\nTheta1: 
      {:0.3f}'.format(theta[0][0], theta[1][0]))
print('Final cost/MSE: {:0.3f}'.format(cost_history[-1]))
 

# Output:
 
Theta0:  3.935,
Theta1:  3.040
Final cost/MSE: 963.651
 

# You can also define loss functions and SGD function in `pytorch`, which is rather simple. Just like:
# Model
 
model = torch.nn.Linear(in_features, out_features)
 

# Loss function

 
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(out, target)
 
# Optimizer
 
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.01, momentum=0.9)
 

