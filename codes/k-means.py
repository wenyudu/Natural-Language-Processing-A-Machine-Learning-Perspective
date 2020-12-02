#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 2-means and 3-means clustering using `Scikit.learn`
# ```
from sklearn.cluster import KMeans
km_cluster = KMeans(n_clusters=2,max_iter=300,n_init=40,
                   init="k-means++",n_jobs=-1)
result = km_cluster.fit_predict(tfidf_matrix)
print("Predicting result for 2-means:",result)
Predicting result for 2-means: [1 1 0 1]

km_cluster = KMeans(n_clusters=3,max_iter=300,n_init=40,
                   init="k-means++",n_jobs=-1)
result = km_cluster.fit_predict(tfidf_matrix)
print("Predicting result for 3-means:",result)
Predicting result for 3-means: [0 0 1 2]
# ```
# > `init`: method for initilization, defaults to "k-means++".
# `n_init`: number of time the k-means algorithm will be run with different centroid seeds.
# `n_jobs`: the number of jobs to use for the computation.

