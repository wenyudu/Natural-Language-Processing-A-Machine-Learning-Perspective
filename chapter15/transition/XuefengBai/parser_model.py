#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple Feed-Forward transition-based parser for Dependency Parsing
"""

import torch
import torch.nn as nn


class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and two hidden layers.
    """

    def __init__(self, embeddings, n_features=36, hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the model.

        @embeddings (ndarray): word embeddings (num_words, embedding_size)
        @n_features (int): number of input features
        @hidden_size (int): number of hidden units
        @n_classes (int): number of output classes
        @dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden_weight = nn.Parameter(
            torch.empty(self.n_features * self.embed_size, self.hidden_size)
        )
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.hidden_to_logits_weight = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        nn.init.uniform_(self.embed_to_hidden_bias)
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        nn.init.uniform_(self.hidden_to_logits_bias)

    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @w (Tensor): input tensor of word indices (batch_size, n_features)
            @x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        x = torch.index_select(self.embeddings, 0, index=w.view(-1, 1).squeeze())
        x = x.view(w.shape[0], -1)
        return x

    def forward(self, w):
        """ 
        @w (Tensor): input tensor of tokens (batch_size, n_features)

        @logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        x = self.embedding_lookup(w)                            # feature representation Eq 15.15
        line_layer = torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias   # feed forward layer Eq 15.16
        h = nn.ReLU(inplace=True)(line_layer)                   # nonlinear activation
        # self.dropout(h)
        logits = torch.matmul(h, self.hidden_to_logits_weight) + self.hidden_to_logits_bias     # Action prediction (un-normalized) Eq 15.17
        return logits