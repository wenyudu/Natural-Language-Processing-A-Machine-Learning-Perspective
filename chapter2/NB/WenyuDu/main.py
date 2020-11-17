import math
import random
from collections import defaultdict
from data import SST
"""
Author: Wenyu Du
Organization: Westlake University
Email: wenyudu@yahoo.com

This is a simple implementation of naive bayes sentiment classification.

Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

class NaiveBayes():
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

    # given class c, calculate the probability of the occurrence of word w with add-one smoothing
    def _get_prob_smoothing(self,c,word):
        total_count = float(sum(self.wc_count[word].values()))
        return (self.wc_count[word][c] + 1) / (self.c_count[c] + total_count)

    def train(self):
        # Step 2: Calculate P(c)
        # Take class "Pros" as an example, assume there are 1000 sents with "Pros" tag in 2000 samples, self.c_count["Pros"]=1000, self.c_prob["Pros"]=0.5
        self.c_count = defaultdict(lambda: 0) # count each label
        for i in self.train_y: self.c_count[i] += 1
        self.c_prob = defaultdict(lambda: 0) # calculate the probability of each label
        for c in self.c_count: self.c_prob[c] = self.c_count[c] / len(self.train_y)
        self.c_type = len(self.c_count) # the class number

        # Step 3: Create Bag of Word (BOW) for training data
        train_X_BOW = set()
        for x in self.train_X:
            for i in x:
                train_X_BOW.add(i)

        # Step 4: Count w/c
        # Assume there are 200 times of the word "happy" in "Pros" sents and 20 times in "Cons" sents,
        # the self.wc_count["happy"]["Pros"] = 200 and self.wc_count["happy"]["Cons"] = 20
        self.wc_count = defaultdict(lambda: defaultdict(lambda: 0))
        for word in train_X_BOW:
            for i, sent in enumerate(self.train_X):
                if word in sent:
                    self.wc_count[word][self.train_y[i]]+=1 # given class c, count the occurrence of word w

        # Step 5: Calculate P(w/c)
        # Calculate the probability of w/c from step 4, in this case,
        # self.wc_prob["happy"]["Pros"] = (200 + 1) / (1000 + 220) and self.wc_prob["happy"]["Cons"] = (20 + 1) / (1000 + 220)
        self.wc_prob = defaultdict(lambda: defaultdict(lambda: 0))
        for word in train_X_BOW:
            for i in range(self.c_type):
                c = list(self.c_count.keys())[i]
                self.wc_prob[word][c] = self._get_prob_smoothing(c,word) # given class c, calculate the probability of word w with add-one smoothing

    def classify(self,sent):
        # Calculate sum(log(P(w/c)))
        # Take "I am sad" as an example, wclog_sum["Pros"] = sum(log(P("I"/"Pros")) + log(P("am"/"Pros")) +log(P("sad"/"Pros")))
        wclog_sum = defaultdict(lambda: 0)
        for i in range(self.c_type):
            for word in sent:
                c = list(self.c_count.keys())[i]
                wclog_sum[c] += math.log(self._get_prob_smoothing(c,word)) # given class c, accumulate the log of the probability of each word w with add-one smoothing

        # Calculate log(P(d,c)) and output the largest value as the prediction
        # cdlog_prob["Pros"] = sum(log(P("I"/"Pros")) + log(P("am"/"Pros")) +log(P("sad"/"Pros"))) + log(P("Pros")),
        # cdlog_prob["Cons"] = sum(log(P("I"/"Cons")) + log(P("am"/"Cons")) + log(P("sad"/"Cons"))) + log(P("Cons")),
        # print the larger value between cdlog_prob["Pros"] and cdlog_prob["Cons"]
        cdlog_prob = defaultdict(lambda: 0)
        for i in range(self.c_type):
            c = list(self.c_count.keys())[i]
            cdlog_prob[c] = wclog_sum[c] + math.log(self.c_prob[c]) # log(P(d,c)) = sum(log(P(w/c))) + log P(c)
        index = list(cdlog_prob.values()).index(max(cdlog_prob.values()))
        return(list(cdlog_prob.keys())[index])

    def test(self):
        for s in self.data.test['sents']:
            result = self.classify(s)
            output = "Positive" if result > 0 else "Negative"
            print(' '.join(s),": ",output)

if __name__ == '__main__':
    data = SST(fine_grained=False)
    nb = NaiveBayes(data,0.5)
    nb.train()
    nb.test()