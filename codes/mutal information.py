#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Calculate pointwise mutual information between words</font>Corpus: apple, new, city, book, dog, like, good, store

 
from collections import Counter
from math import log
 
# Generate bigrams from data
 
def gen_bigrams(data, window_size=5):
    for idx in range(len(data)):
        window = data[idx: idx + window_size]
        
        if len(window) < 2:
            break
            
        w = window[0]
        for next_word in window[1:]:
            yield (w, next_word)
 

 
# Construct the vocabulary, count word frequency
 
def construct_vocab(data):
    vocab = Counter()
    for (w1, w2) in gen_bigrams(data, window_size=5):
        vocab.update([w1, w2, (w1, w2)])
    return vocab
 

# Construct function to calculate pointwise mutual information

 
def calc_pmi(vocab):
    det = sum(vocab.values())
    for (w1, w2) in filter(lambda el: 
                           isinstance(el, tuple), vocab):
        p_a, p_b = float(vocab[w1]), float(vocab[w2])
        p_ab = float(vocab[(w1, w2)])
        yield (w1, w2, log((det * p_ab) / (p_a * p_b), 2))
 
# `filter` filters the given sequence with the help of a function that tests each element in the sequence to be true or not

 
corpus = ["apple", "new", "city", "book", 
          "dog", "like", "good", "store"]
vocab = construct_vocab(corpus)
for (w1, w2, pmi) in calc_pmi(vocab):
    print("{}_{}: {:.3f}".format(w1, w2, pmi))
 
# Output:
``` 
apple_new: 1.722  apple_city: 1.459  apple_book: 1.237 
apple_dog: 1.237  new_city: 1.138  new_book: 0.915 
new_dog: 0.915  new_like: 1.138  city_book: 0.652
city_dog: 0.652  city_like: 0.874  city_good: 1.138
book_dog: 0.430  book_like: 0.652  book_good: 0.915
book_store: 1.237  dog_like: 0.652  dog_good: 0.915
dog_store: 1.237  like_good: 1.138  like_store: 1.459
good_store: 1.722
``` 

