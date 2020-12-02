#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SVM text classification (sentiment analysis) with `scikit.learn`

# Dataset:  [Cornell Natural Language Processing Group](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz).
# * Importing Libraries
# * Importing The dataset
# * Text Preprocessing
# * Converting Text to vectors
# * Split training and Test Sets
# * Training Text Classification Model and Predicting Sentiment using SVM


# Import libraries and dataset
#```
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords 

movie_data = load_files(r"home/Downloads/dataset")  
X, y = movie_data.data, movie_data.target  

## Text preprocessing
# ```python
documents = []
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)

## Convert text to vectors and calculate TFIDF
```
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(max_features=1500, min_df=5, 
          max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()  

from sklearn.feature_extraction.text import TfidfTransformer  
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()  

from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, 
          max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(documents).toarray()  

## Split training and testing sets and build SVM classifier

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                           test_size=0.2, random_state=0)
                           
From sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

