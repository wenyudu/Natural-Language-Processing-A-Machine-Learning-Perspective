#!/usr/bin/env python3 -u
# Copyright (c) Westlake University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Created by Guangsheng Bao on 2020/10/29.

import io
import codecs
import numpy as np

# load data for demonstration, each line contains: <en words>\t<zh words>
def load_data():
    with codecs.open('data/parallel_text.en-zh.txt', 'r', 'utf-8') as fin:
        lines = [line.strip().split('\t') for line in fin.readlines()]
        lines = [(en.split(), zh.split()) for en, zh in lines]
    return lines

# Chapter 6 - Hidden Variables
# Algorithm 6.3: Word alignment.
# D - data
def word_align(D, seed=0):
    def _converged(param, last_param):
        if last_param is None:
            return False
        loss = np.mean(np.square(probX_givenY - last_param))
        print('Loss:', loss)
        return loss <= 1e-6

    # build vocab
    target_vocab = set()
    source_vocab = set()
    for X, Y in D:
        target_vocab.update(X)
        source_vocab.update(Y)
    target_vocab = dict((w, i) for i, w in enumerate(target_vocab))
    source_vocab = dict((w, i) for i, w in enumerate(source_vocab))
    print('Target vocab size:', len(target_vocab))
    print('Source vocab size:', len(source_vocab))
    D = [([target_vocab[x] for x in X], [source_vocab[y] for y in Y]) for X, Y in D]
    # train
    np.random.seed(seed)
    probX_givenY = np.random.random([len(target_vocab), len(source_vocab)])
    last_param = None
    while not _converged(probX_givenY, last_param):
        last_param = probX_givenY.copy()
        # Expectation step
        countX_givenY = np.zeros([len(target_vocab), len(source_vocab)], dtype=np.float)
        countY = np.zeros([len(source_vocab)], dtype=np.float)
        for X, Y in D:
            senttotalX = np.zeros([len(X)], dtype=np.float)
            for i, x in enumerate(X):
                for y in Y:
                    senttotalX[i] += probX_givenY[x, y]
            for i, x in enumerate(X):
                for y in Y:
                    countX_givenY[x, y] += probX_givenY[x, y] / senttotalX[i]
                    countY[y] += probX_givenY[x, y] / senttotalX[i]
        # Maximization step
        for x in target_vocab.values():
            for y in source_vocab.values():
                probX_givenY[x, y] = countX_givenY[x, y] / countY[y]
    return probX_givenY, source_vocab, target_vocab



if __name__ == '__main__':
    data = load_data()

    # test: word align
    print('------')
    print('Algorithm 6.3: Word alignment.')
    print('Test with predefined data: parallel_text.en-zh.txt')
    print('Observed samples:', len(data))
    # train model: P(x|y)
    probX_givenY, source_vocab, target_vocab = word_align(data, seed=1)
    # print top 10 aligned words
    print('Top 10 mappings:')
    source_vocab = dict((source_vocab[key], key) for key in source_vocab)
    target_vocab = dict((target_vocab[key], key) for key in target_vocab)
    nx, ny = probX_givenY.shape
    prob = probX_givenY.reshape([-1])
    top = np.argsort(-prob)[:10]
    for idx in top:
        i = idx // ny
        j = idx % ny
        print('P(%s|%s)=%s' % (target_vocab[i], source_vocab[j], probX_givenY[i, j]))

