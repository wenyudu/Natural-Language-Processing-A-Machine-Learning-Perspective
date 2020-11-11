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

# load data for demonstration, each line contains: <word> <embedding>
def load_data():
    with codecs.open('data/chinese_embedding.txt', 'r', 'utf-8') as fin:
        lines = [line.strip().split() for line in fin.readlines()]
        lines = [(line[0], [float(v) for v in line[1:]]) for line in lines]
    return lines

# Chapter 6 - Hidden Variables
# Algorithm 6.1: K-means as a “hard” EM algorithm.
# O - observed data
# K - number of clusters
def kmeans_hardem(O, K, seed=0):
    np.random.seed(seed)
    N = O.shape[0]  # number of samples
    D = O.shape[1]  # dimension of the vector
    H = np.zeros([N, K], dtype=np.int)  # hidden variable
    C = O[np.random.randint(0, N, K)]  # model \theta
    last_H = None
    while last_H is None or np.any(H != last_H):
        last_H = H
        # Expectation step
        dist = np.linalg.norm(O.reshape([N, 1, D]) - C.reshape([1, K, D]), axis=2)
        k = dist.argmin(axis=1)
        H = np.zeros_like(H)
        H[np.arange(N), k] = 1
        # Maximization step
        C = np.sum(H.reshape(N, K, 1) * O.reshape(N, 1, D), axis=0) / (np.sum(H, axis=0).reshape(K, 1) + 1e-9)
    return C

# Chapter 6 - Hidden Variables
# Algorithm 6.2: Expectation maximisation.
# O - observed data
# K - number of clusters
def kmeans_em(O, K, seed=0):
    np.random.seed(seed)
    N = O.shape[0]  # number of samples
    D = O.shape[1]  # dimension of the vector
    H = np.zeros([N, K], dtype=np.float)  # hidden variable
    C = O[np.random.randint(0, N, K)]  # model \theta
    last_H = None
    while last_H is None or np.sum(np.square(H - last_H)) > 1e-6:
        last_H = H
        # Expectation step
        dist = np.linalg.norm(O.reshape([N, 1, D]) - C.reshape([1, K, D]), axis=2)
        dist = np.exp(- dist**2 / 2)  # assume normal Gaussian distribution for each cluster
        H = dist / np.sum(dist, axis=1, keepdims=True)
        # Maximization step
        C = np.sum(H.reshape(N, K, 1) * O.reshape(N, 1, D), axis=0) / (np.sum(H, axis=0).reshape(K, 1) + 1e-9)
    return C


def classify(C, v):
    D = v.shape[0]
    dist = np.linalg.norm(v.reshape([1, D]) - C, axis=1)
    k = dist.argmin(axis=0)
    return k

if __name__ == '__main__':
    data = load_data()
    O = np.array([d[1] for d in data])  # observed data
    K = 20  # number of clusters

    # test1: hard EM
    print('------')
    print('Algorithm 6.1: K-means as a “hard” EM algorithm.')
    print('Test with predefined data: chinese_embedding.txt')
    print('Observed samples:', len(data))
    print('Number of clusters:', K)
    # train model C
    C = kmeans_hardem(O, K, seed=1)
    # predict
    results = [[] for _ in range(K)]
    for w, v in data:
        v = np.array(v)
        k = classify(C, v)
        results[k].append(w)
    # print results
    print('Result clusters:', results)

    # test2: EM
    print('------')
    print('Algorithm 6.2: Expectation maximisation.')
    print('Test with predefined data: chinese_embedding.txt')
    print('Observed samples:', len(data))
    print('Number of clusters:', K)
    # train model C
    C = kmeans_em(O, K, seed=1)
    # predict
    results = [[] for _ in range(K)]
    for w, v in data:
        v = np.array(v)
        k = classify(C, v)
        results[k].append(w)
    # print results
    print('Result clusters:', results)


