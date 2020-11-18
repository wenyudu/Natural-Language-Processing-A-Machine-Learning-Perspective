#!/usr/bin/env python
# coding: utf-8

# # Siamese Network for Semantic Similarity
#  Classification of items based on their similarity is one of the major challenge of Machine Learning and deep learning problems. 
#  
#  A Siamese Neural Network is a class of neural network structures that contain two or more identical sub networks. They have the same configuration with the same parameters and weights. Parameter update is mirrored across both sub networks. It is used to find the similarity of the inputs by comparing its feature vectors.

# ## Code Walk Through
# Dataset: [Quora Question Pair Dataset](https://www.kaggle.com/c/quora-question-pairs/data)
# 
# Requirements: 
# * Python 3.8
# * TensorFlow
# * Scikit-Learn
# * Gensim
# * NLTK

# ### Dataset

# In[ ]:


import pandas as pd

df_train = pd.read_csv("train.csv/train.csv")
df_train.head()


# * id: The ID of the training set of a pair
# * qid1, qid2: Unique ID of the question
# * question1: Text for question one
# * question2: Text for question two
# * is_duplicate: 1 if question1 and question2 have the same meaning or elso 0

# ### Import all the necessary packages

# In[ ]:


from time import time
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# ### Global variables

# In[ ]:


# File path
train_csv = './train.csv/train.csv'
test_csv = './test.csv/test.csv'
embedding_file = './GoogleNews-vectors-negative300.bin.gz'
model_saving_dir = './models/'


# In[ ]:


import nltk
nltk.download('stopwords')


# ### Create embedding matrix

# In[ ]:


# Load training and test set
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

stops = set(stopwords.words('english'))

def text_to_word_list(text):
    #Preprocessing and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'_-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d" , " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/" , " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\=" , " = ", text)
    text = re.sub(r"\-" , " - ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j K", "jk", text)
    text = re.sub(r"\s{2, }", " ", text)

    text = text.split()

    return text


# In[ ]:


# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']

word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

questions_cols = ['question1', 'question2']


# In[ ]:


# iteration over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        # iterate through the text of both questions of the row
        for question in questions_cols:
            q2n = []  # question numbers representation
            for word in text_to_word_list(row[question]):
                # check for unwanted words
                if word in stops and word not in word2vec.vocab:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, question, q2n)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0


# In[ ]:


# building the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


# ### Prepare training and validation data

# In[ ]:


max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
train_df.question2.map(lambda x: len(x).max()),
test_df.question1.map(lambda x: len(x).max()),
test_df.question2.map(lambda x: len(x).max()))

# split to train validation
validation_size = 40000
train_size = len(train_df) - validation_size

X = train_df[question_cols]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size)

# split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}

# Convert labels to their numpy representation
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# ### Building the model

# In[ ]:


# Model variables
n_hidden = 50
gradient_clippping_norm = 1.25
batch_size = 64
n_epoch = 25

def exponent_neg_manhattan_distance(left, right):
    # helper function for the similarity estimate of the LSTMs outputs
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# In[ ]:


# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], 
input_length=max_seq_lenght, trainable=False)


# In[ ]:


# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)


# In[ ]:


# siamese lstm network
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)


# In[ ]:


# calculate the distance
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0],x[1],
output_shape=lambda x: (x[0][0], 1))([left_output, right_output]))


# In[ ]:


# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# adadelta optimizer, wiht gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clippping_norm)

malstm.compile(loss='mean_square_error', optimizer=optimizer, metrics=['accuracy'])


# In[ ]:


# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], 
Y_train, batch_size=batch_size, nb_epoch=n_epoch,
validation_data=([X_validation['left'], X_validation['right']], Y_validation))
print("Training time finished. \n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time))


# ### Plot the results

# In[ ]:


# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], lco='upper left')
plt.show()


# In[ ]:


# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# 
