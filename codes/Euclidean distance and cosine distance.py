#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Calculate Euclidean distance and cosine distance using pytorch

# For our documents $[d_1,d_2,d_3,d_4]$, calculate their similarity using `torch.dist` and  `torch.cosine_similarity`

# Compare the distance between $d_1$ and $d_2$, to the distance between $d_1$ and $d_4$, what you can see?
# 1. Assign the TF-IDF vector representation to the target docuents

# ```
# ...
d1 = tfidf[0]
d2 = tfidf[1]
d4 = tfidf[3]
# ```

# 2. Calculate Euclidean distance using the `pytorch` module
# ```python
d1_d2 = torch.dist(d1, d2)
d1_d4 = torch.dist(d1, d4)
# ```
 Result:
 ```python
print(d1_d2, d1_d4)

tensor(0.5242) tensor(0.6885)
# ```
# 2. Calculate cosine distance using the `pytoch` module
# ```python
d1_d2 = torch.cosine_similarity(d1, d2, dim=0)
d1_d4 = torch.cosine_similarity(d1, d4, dim=0)
# ```
# Result:
# ```python
print(d1_d2, d1_d4)

tensor(0.1079) tensor(0.0228)
# ```

