#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Text classification (sentiment analysis) with `scikit.learn` with Perceptron algorithm


 from sklearn.linear_model import Perceptron
 ppn = Perceptron(n_iter = 40, eta0 + 0.1, random_state = 1)
 ppn.fit(X_train,y_train)
 y_pred = ppn.predict(X_test)

# > `n_iter`: (int) the number of passes over the training data, defaults to 5
# > `eta0`: (double) constant by which the updates are multiplied, defaults to 1.
# > `random_state`: whether or not the training data should be shuffled after each epoch. None default.

