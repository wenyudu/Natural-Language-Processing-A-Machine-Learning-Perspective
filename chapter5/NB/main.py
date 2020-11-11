import math
import random
from collections import defaultdict
from data import SST
"""
Author: Wenyu Du
Organization: Westlake University
Email: wenyudu@yahoo.com

This is a simple implementation of Pointwise Mutual Information (PMI).

Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

class PMI():
    def __init__(self,data,split=1.0):
        # Step 1 : Load data and split sets (if required)

        # load data X and label y
        X = data.train["sents"]
        y = data.train["labels"]
        self.data = data

        # shuffle data and label with the same seed
        random.shuffle(X,lambda: 0.1) # shuffle data X
        random.shuffle(y,lambda: 0.1) # shuffle label y

        # due to the size of data, only use part of the corpus for training if needed
        split_point = int(len(X) * split)
        self.train_X = X[:split_point]
        self.train_y = y[:split_point]
        self.train_size = len(self.train_X)

    def calculate(self,w,seed):
        # Step 2: calculate P(w) and P(seed)
        w_count = 0
        seed_count = 0
        for i, sent in enumerate(self.train_X):
            if w in sent:
                w_count+=1 # count the occurrence of word w
            if seed in sent:
                seed_count+=1 # count the occurrence of word seed
        p_w = w_count / self.train_size
        p_seed = seed_count / self.train_size

        # Step 3: calculate P(w,seed)
        w_seed_count = 0
        for i, sent in enumerate(self.train_X):
            if (w in sent) and (seed in sent):
                w_seed_count+=1 # count the occurrence of word seed
        p_w_seed = w_seed_count / self.train_size / self.train_size

        # Step 4: calculate PMI
        value = math.log2(p_w_seed/(p_w*p_seed))

        return value

if __name__ == '__main__':
    data = SST(fine_grained=False)
    pmi = PMI(data,0.5)
    pmi_1=pmi.calculate("movie","good")
    pmi_2=pmi.calculate("movie","bad")
    lex_w = pmi_1-pmi_2