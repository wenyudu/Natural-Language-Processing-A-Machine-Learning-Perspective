import random
import numpy as np
from data import SQuAD_docs
"""
Author: Cunxiang Wang
Organization: Westlake University
Email: wangcunxiang@westlake.edu.cn

This is a simple implementation of Latent Dirichlet Allocation model.

Data: 2000 docs from SQuAD dataset (only docs are used). 
"""

class LDA():
    def __init__(self, data, topic_number, alpha, beta, epochs):
        self.vocab_size = data.vocab_size
        self.data = data.data # We only use train data
        self.dict =data.dict
        self.topic_number = topic_number
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.z, self.n_doc_topic, self.n_doc_topic_sum, self.c_topic_word, self.c_topic_word_sum \
            = self.initialization(self.data, self.topic_number, self.vocab_size, self.dict)

    def initialization(self, data, topic_number, vocab_size, dict):
        '''
        The Initialisation part of Algorithm 12.2
        :param data: [[word]], len(data)=doc_number, len(data[i])=length of doc_i
        :param topic_number: int, topic_number
        :param vocab_size: int, vocab_size
        :param dict: {word:i}, dictionary
        :return:z, n_doc_topic, n_doc_topic_sum, c_topic_word, c_topic_word_sum
        '''
        z = []
        n_doc_topic = np.zeros((len(data), topic_number), dtype=float)
        n_doc_topic_sum = np.zeros(len(data), dtype=float)
        c_topic_word = np.zeros((topic_number, vocab_size), dtype=float)
        c_topic_word_sum = np.zeros(topic_number, dtype=float)

        for d, doc in enumerate(data):
            z.append([])
            for j, word in enumerate(doc):
                k = random.randint(0, topic_number-1)
                z[d].append(k)
                i_word = dict[word]
                c_topic_word[k, i_word] += 1
                c_topic_word_sum[k] += 1
                n_doc_topic[d, k] += 1
                n_doc_topic_sum[d] += 1
        print("Initialization Done")
        return np.array(z), n_doc_topic, n_doc_topic_sum, c_topic_word, c_topic_word_sum

    def run(self):
        data = self.data
        dict = self.dict
        alpha = self.alpha
        beta = self.beta
        epochs = self.epochs
        z = self.z # [[int]] not np.array
        topic_number = self.topic_number
        n_doc_topic = self.n_doc_topic
        n_doc_topic_sum = self.n_doc_topic_sum
        c_topic_word = self.c_topic_word
        c_topic_word_sum = self.c_topic_word_sum
        for epoch in range(epochs):
            for d, doc in enumerate(data):
                for j, word in enumerate(doc):
                    k = z[d][j]
                    i_word = dict[word]
                    c_topic_word[k, i_word] -= 1
                    c_topic_word_sum[k] -= 1
                    n_doc_topic[d, k] -= 1
                    n_doc_topic_sum[d] -= 1
                    k_new = self.cal_z_k(d, i_word, alpha, beta, topic_number, n_doc_topic, n_doc_topic_sum, c_topic_word, c_topic_word_sum)
                    z[d][j] = k_new
                    c_topic_word[k_new, i_word] += 1
                    c_topic_word_sum[k_new] += 1
                    n_doc_topic[d, k_new] += 1
                    n_doc_topic_sum[d] += 1
            print("Epoch {} done".format(epoch))

        return z, self.get_theta(alpha, n_doc_topic), self.get_phi(beta, c_topic_word)



    def cal_z_k(self, d, i_word, alpha, beta, topic_number, n_doc_topic, n_doc_topic_sum, c_topic_word, c_topic_word_sum):
        '''
        Calculating z_j according to formula 12.44
        :return: the k which makes the highest score
        '''
        scores = np.array([0.0 for i in range(topic_number)])
        left_bottom = n_doc_topic_sum[d] + alpha.sum()  # float
        for k in range(topic_number):
            left_top = n_doc_topic[d, k] \
                       + alpha[k]
            right_top = c_topic_word[k, i_word] \
                        + beta[i_word]
            right_bottom = c_topic_word_sum[k] + beta.sum()
            scores[k] = 1000 * left_top / left_bottom * right_top / right_bottom

        return np.where(scores==np.max(scores))

    def get_theta(self, alpha, n_doc_topic):
        theta = np.zeros((len(n_doc_topic), len(alpha)), dtype=float) # np.array [docs number, topic_number]
        for d in range(len(n_doc_topic)): # length = docs number
            bottom = n_doc_topic[d].sum()+alpha.sum()
            for j in range(len(alpha)): # length = topic_number
                theta[d, j] = (n_doc_topic[d, j] + alpha[j]) / bottom
        return theta

    def get_phi(self, beta, c_topic_word):
        phi = np.zeros((len(c_topic_word), len(beta)), dtype=float) # np.array [topic_number, vocab_number]
        for k in range(len(c_topic_word)): # length = topic_number
            bottom = c_topic_word[k].sum()+beta.sum()
            for j in range(len(beta)): # length = vocab_number
                phi[k, j] = (c_topic_word[k, j] + beta[j]) / bottom
        return phi

if __name__ == '__main__':
    data = SQuAD_docs(path="/home/sean/Desktop/chapter12/data/SQuAD_docs")
    topic_number = 20
    alpha = np.array([0.1 for i in range(topic_number)])
    beta = np.array([0.1 for i in range(len(data.dict))])
    epochs = 20
    print("Topic number is {}; Epoch is {}".format(topic_number, epochs))
    lda = LDA(data, topic_number, alpha, beta, epochs)
    z, theta, phi = lda.run()
    dict_ = {v : k for k,v in lda.dict.items()}
    top_k = 10
    for i in range(topic_number):
        order = np.argsort(phi[i])
        print("The tok {} words related to topic {}:".format(top_k, i))
        tmp = [print(dict_[order[j]], end='\t') for j in range(top_k)]
        print()