#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Count-based document representation.
# 1. Import python modules `pytorch`, `collections` and `math`
# ```python
import torch
from collections import Counter
import math
# ```
# 2. Load dataset and define the stop-words
# ```python
documents = ["Tim bought a book .",
            "Tim is reading a book .",
            "ah , Tim is Tim .",
            "I saw a boy reading a book ."]
            
stop_words = ['a', '.', ',']
# ```

# 3. Clean stop-words and count word frequency
# ```python
clean_docs = []
word_count = Counter()
for doc in documents:
    word_count.update([wd for wd in doc.strip().split(' ') 
                      if wd not in stop_words])
    clean_docs.append([wd for wd in doc.strip().split(' ') 
                      if wd not in stop_words])
# ```
# 4. Build up the vocabulary
# ```python
vocab = [word for word in word_count.keys()]
# ```


# Check the loaded data
# ```python
print(clean_docs)

[['Tim', 'bought', 'book'],
['Tim', 'is', 'reading', 'book'],
['ah', 'Tim', 'is', 'Tim'],
['I', 'saw', 'boy', 'reading', 'book']]
# ```
# Word count
# ```python
print(word_count)

Counter({'Tim': 4, 'book': 3, 'is': 2, 'reading': 2,
        'bought': 1, 'ah': 1, 'I': 1, 'saw': 1, 'boy': 1})
```
# Vocabulary
# ```python
print(vocab)

['Tim', 'bought', 'book', 'is',
'reading', 'ah', 'I', 'saw', 'boy']
# ```
# 6. Count-based document representation
# ```python
count_vec = torch.zeros(len(clean_docs), len(vocab))
for i in range(len(clean_docs)):
    for j in range(len(vocab)):
        count = 0
        for word in clean_docs[i]:
            if word == vocab[j]:
                count += 1
        count_vec[i][j] = count
# ```
# Result:
# ```python
print(count_vec)

tensor([[1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 1., 1., 1., 0., 0., 0., 0.],
        [2., 0., 0., 1., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 1., 1., 1.]])
# ```

# TF-IDF vectors calculation using python.

# 7. Count the number of documents that contain a certain vocabulary word
# ```python
doc_count = torch.ones(1, len(vocab))

for i in range(len(vocab)):
    freq = 0
    for doc in clean_docs:
        if vocab[i] in doc:
            freq += 1
    doc_count[0][i] = freq
# ```
# ```
print(doc_count)

tensor([[3., 1., 3., 2., 2., 1., 1., 1., 1.]])
# ```

# 8. Count the vocabulary words in each document

# ```python
doc_len = torch.zeros(len(clean_docs), 1)
for i in range(len(clean_docs)):
    doc_len[i][0] = len(clean_docs[i])
# ```
# 9. Calculate the term frequency
# ```python
tf = count_vec/doc_len
# ```
# Result:
# ```python
print(tf)

tensor([[0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.0000, 0.2500, 0.2500, 0.2500, 
        0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000, 0.2500, 0.0000, 
        0.2500, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2000, 0.0000, 0.2000, 
        0.0000, 0.2000, 0.2000, 0.2000]])
# ```


# 10. Calculate the inverted document frequency
# ```python
idf = torch.log(torch.ones(len(documents), 
                len(vocab))*len(documents)/doc_count)
# ```
# Result:
# ```
print(idf)

tensor([[0.2877, 1.3863, 0.2877, 0.6931, 0.6931,
        1.3863, 1.3863, 1.3863, 1.3863],
        [0.2877, 1.3863, 0.2877, 0.6931, 0.6931,
        1.3863, 1.3863, 1.3863, 1.3863],
        [0.2877, 1.3863, 0.2877, 0.6931, 0.6931,
        1.3863, 1.3863, 1.3863, 1.3863],
        [0.2877, 1.3863, 0.2877, 0.6931, 0.6931,
        1.3863, 1.3863, 1.3863, 1.3863]])
# ```



# 11. TF-IDF vector document representation
# ```python
tfidf = tf*idf
# ```
# ```
print(tfidf)

tensor([[0.0959, 0.4621, 0.0959, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000],
        [0.0719, 0.0000, 0.0719, 0.1733, 0.1733, 
        0.0000, 0.0000, 0.0000, 0.0000],
        [0.1438, 0.0000, 0.0000, 0.1733, 0.0000, 
        0.3466, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0575, 0.0000, 0.1386, 
        0.0000, 0.2773, 0.2773, 0.2773]])
# ```

