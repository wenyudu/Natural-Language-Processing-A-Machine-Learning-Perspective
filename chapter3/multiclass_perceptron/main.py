import random
import torch
import torch.nn as nn
import torch.optim as optim
from data import SST

"""
Author: Wenyu Du
Organization: Westlake University
Email: wenyudu@yahoo.com

This is a simple implementation of the multi-class SVM classifier that optimised by SGD.

Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

class Multiclass_Perceptron():
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

        # the dictionary of the vocabulary and the class_name
        self.voc_dict = self._get_voc_dict() # vocabulary dictionary
        self.voc_size = len(self.voc_dict)
        self.class_dict = list(set(self.train_y)) # class dictionary
        self.class_size = len(self.class_dict)

        # hyperparameters for training
        self.epoch_number = 20
        self.lr = 0.3 # learning rate
        self.model = nn.Linear(self.voc_size * self.class_size, 1) # w, b in pytorch

    # get a vocabulary dictionary for generating feature vectors
    def _get_voc_dict(self):
        train_X_BOW = set() # a set for storing tokens (words)
        for x in self.train_X:
            for i in x:
                train_X_BOW.add(i) # add each token into the set
        return list(train_X_BOW) # return as a list

    # generate feature vectors for sentences with classes
    def _get_feature_vec(self,string_sents):
        multiclass_X = []
        for class_id in range(self.class_size):
            X = torch.zeros(len(string_sents),self.voc_size*self.class_size) # initialize feature vector with zeros
            for r,x in enumerate(string_sents):
                for i in x:
                    c = self.voc_dict.index(i)
                    X[r][c+class_id*self.voc_size] += 1 # add one when the sentence r contains token c with class class_id
            multiclass_X.append(X)
        return multiclass_X

    # the SGD optimiser
    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr)

    # run one optimizer step
    def _optimizer_step(self,optimizer,loss):
        loss.backward()
        optimizer.step()
        return float(loss)

    def train(self):
        multiclass_X = self._get_feature_vec(self.train_X)
        optimizer = self._optimizer()
        self.model.train()

        for epoch in range(self.epoch_number):
            sum_loss = 0
            for i in range(self.train_size):
                y = self.train_y[i]
                optimizer.zero_grad()

                # calculate outputs for all classes c
                outputs = []
                for j in range(self.class_size):
                    outputs.append(self.model(torch.FloatTensor(multiclass_X[j][i])).squeeze()) # wv+b for all class in pytorch
                max_output = max(outputs)
                max_output_idx = outputs.index(max_output)

                ###### loss function ######
                if max_output_idx == y: # condition of updating parameters of w and b
                    continue
                loss = torch.clamp(-outputs[y] + outputs[max_output_idx], min=0) # loss function
                sum_loss += self._optimizer_step(optimizer, loss)  # sum of loss in one epoch
                #############################
            # self.test()
            print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / self.train_size))

    def test(self):
        self.model.eval()
        X = self._get_feature_vec(self.data.test["sents"])
        for i in range(X[0].shape[0]):
            outputs=[]
            for j in range(self.class_size):
                outputs.append(self.model(torch.FloatTensor(X[j][i])).squeeze()) # dot product for all class in pytorch
            max_output = max(outputs)
            max_output_idx = outputs.index(max_output) # the max output as the predicted class c
            c = self.class_dict[max_output_idx]
            if c == 0:
                c_string = "Very negative"
            elif c == 1:
                c_string = "Negative"
            elif c == 2:
                c_string = "Neutural"
            elif c == 3:
                c_string = "Positive"
            else:
                assert c == 4
                c_string = "Very positive"

            print(' '.join(self.data.test["sents"][i]),": ",c_string)

if __name__ == '__main__':
    data = SST(fine_grained=True)
    classifier = Multiclass_Perceptron(data,0.2)
    classifier.train()
    classifier.test()