import random
import torch
import torch.nn as nn
import torch.optim as optim
from data import SST

"""
Author: Wenyu Du
Organization: Westlake University
Email: wenyudu@yahoo.com

This is a simple implementation of the binary log linear classifier that optimised by SGD.

Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

class Binary_log_linear():
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
        self.train_X = X[:split_point] # data used for training
        self.train_y = y[:split_point] # labels used for training
        self.train_size = len(self.train_X)

        # the dictionary of the vocabulary
        self.voc_dict = self._get_voc_dict() # vocabulary dictionary
        self.voc_size = len(self.voc_dict)

        # hyperparameters for training
        self.epoch_number = 30
        self.lr = 0.3 # learning rate
        self.model = nn.Linear(self.voc_size, 1) # w, b in pytorch

    # get a vocabulary dictionary for generating feature vectors
    def _get_voc_dict(self):
        train_X_BOW = set() # a set for storing tokens (words)
        for x in self.train_X:
            for i in x:
                train_X_BOW.add(i) # add each token into the set
        return list(train_X_BOW) # return as a list

    # generate feature vectors for sentences
    def _get_feature_vec(self, string_sents):
        X = torch.zeros(len(string_sents), self.voc_size) # initialize feature vector with zeros
        for r, x in enumerate(string_sents):
            for i in x:
                c = self.voc_dict.index(i)
                X[r][c] += 1 # add one when the sentence r contains token c
        return X

    # load the training data as pytorch tensor
    def _tensorize_data(self):
        X = torch.FloatTensor(self._get_feature_vec(self.train_X))
        Y = torch.FloatTensor(self.train_y)
        return X,Y

    # the SGD optimiser
    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr)

    # run one optimizer step
    def _optimizer_step(self,optimizer,loss):
        loss.backward()
        optimizer.step()
        return float(loss)

    def train(self):
        X, Y = self._tensorize_data()
        optimizer = self._optimizer()
        self.model.train()

        for epoch in range(self.epoch_number):
            for i in range(self.train_size):
                optimizer.zero_grad()

                ###### loss function ######
                output = self.model(X[i]).squeeze() # wv+b in pytorch
                P_positive = torch.sigmoid(output) # the probability of y=+1

                with torch.no_grad():
                    if Y[i] == 1:
                        self.model.weight -= self.lr * (P_positive-1) * X[i]
                    else:
                        self.model.weight -= self.lr * (P_positive) * X[i]

    def test(self):
        self.model.eval()
        X = self._get_feature_vec(self.data.test["sents"])
        for i,s in enumerate(self.data.test["sents"]):
            output = "Positive" if torch.sigmoid(self.model(X[i]).squeeze().data) > 0.5 else "Negative" # dot product in pytorch
            print('*' * 20)
            print(' '.join(s),": ",output)

if __name__ == '__main__':
    data = SST(fine_grained=False)
    classifier = Binary_log_linear(data)
    classifier.train()
    classifier.test()