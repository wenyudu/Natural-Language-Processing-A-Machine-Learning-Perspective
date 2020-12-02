#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Document classification with a MaxEnt model</font>
# Using `movie_reviews` dataset in NLTK library
# $*$ each review is labelled either as positive or negative
# 1. prepare list of labelled documents

 
from nltk import *
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in moive_reviews.fileids(category)]
random.shuffle(documents)
 

 
# 2. use the bag-of-words hypothesis, select the 2000 most frequent words
 
all_words = nltk.FreqDist(w.lower() 
                          for w in movie_review.words())
word_features = all_words.keys()[:2000]
 
#  `FreqDist`: a frequency distribution for the input data, returns a dictionary where `key` is the elements and `value` is frequency


# 3. Define selected features in documents
 
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = 
            (word in document_words)
    return features
 
# `set`: convert any of the iterable to the distinct element and sorted sequence of iterable elements
# `contains`: checks for a key present or not in a dictionary

 
# 4. define the MaxEnt classifier
 
featuresets = [(document_features(d), c) 
                for (d, c) in documents]
    
train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)

accuracy = nltk.classifiy.accuracy(classifier, test_set)
classifier.show_most_informative_features(5

