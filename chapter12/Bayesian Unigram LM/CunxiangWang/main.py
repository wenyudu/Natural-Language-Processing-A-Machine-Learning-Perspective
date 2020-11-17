from scipy.special import loggamma
from math import pow, log
from data import PTB

"""
Author: Cunxiang Wang
Organization: Westlake University
Email: wangcunxiang@westlake.edu.cn

This is a simple implementation of Bayesian unigram language model.

Data: The Penn Treebank (PTB) contains 42,068/3,077/3,761 sentences in train/valid/test set. You can access this dataset at https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data

"""

class BayesianUnigramLM():
    def __init__(self, data):

        self.vocab_size = data.vocab_size
        self.train = data.train
        self.valid = data.valid
        self.test = data.test

    def cal_corpus_perplexity(self, log_likelihood, corpus):
        '''
        :param log_likelihood: ln_likelihood
        :return: the perplexity
        '''
        corpus_length = len(corpus)
        return pow(2, -log_likelihood/corpus_length)

    def cal_corpus_log_likelihood_2_2(self, n, theta):
        '''
        Calculate the log_likelihood of one corpus given a theta (Based on the first formula in section 12.2.5, which is also formula 2.2 )
        :param n: [float list] counts of words in this corpus
        :param theta: [float list] len(theta)==len(n)==self.vocab_size and each element of alpha is the same
        :return: the log_likelihood of one corpus given a theta
        '''

        assert len(n)==len(theta)
        assert len(n)==self.vocab_size

        # However, we cannot calculate formula the formula directly (too large). So, we log it.
        log_p_D_theta = 0.0
        for i in range(self.vocab_size):
            log_p_D_theta += n[i] * log(theta[i])
        return log_p_D_theta

    def cal_E_theta(self, n, alpha):
        '''
        Calculate the expectation of theta (Based on formula 12.28)
        :param n: [float list] counts of words in this corpus
        :param alpha: [float list] len(alpha)==len(n)==self.vocab_size and each element of alpha is the same
        :return: the expectation of theta
        '''
        assert len(n) == len(alpha)
        assert len(n) == self.vocab_size
        sum_n_alpha = 0.
        for i in range(self.vocab_size):
            sum_n_alpha += n[i] + alpha[i]
        E_theta = [(n[i]+alpha[i])/sum_n_alpha for i in range(self.vocab_size)]
        return E_theta

    def cal_corpus_log_likelihood_12_30(self, n, alpha):
        '''
        Calculate the log_likelihood of one corpus given a alpha (Based on formula 12.30)
        :param n: [float list] counts of words in this corpus
        :param alpha: [float list] len(alpha)==len(n)==self.vocab_size and each element of alpha is the same
        :return: the log_likelihood of one corpus given a alpha
        '''

        # However, we cannot calculate formula 12.30 directly. So, we log it first,
        # Making  log(P(D|\vec{\alpha}))=log(\Delta(\vec{n}+\vec{\alpha}))-log(\Delta(\vec{\alpha}))
        # then, we pass the two items in cal_Delta_alpha() to calculate them, respectively
        assert len(n)==len(alpha)
        assert len(n)==self.vocab_size
        n_alpha = [n[i]+alpha[i] for i in range(len(n))]
        return self.cal_Delta_alpha(n_alpha)-self.cal_Delta_alpha(alpha)


    def cal_Delta_alpha(self, x):
        '''
        Calculate the log(\Delta(\vec{x})) (Based on formula 12.16)
        :param x: [float list]
        :return: log(\Delta(\vec{x}))
        '''

        # According to formula 12.16,
        # log(\Delta(\vec{x}))=\sum_{k=1}^{K}log(\Gamma(x_{k}))-log(\Gamma(\sum_{k=1}^{K}x_{k}))
        # So, we introduce scipy.special.loggamma to calculate ln(\Gamma(x))
        left = 0.0 # \sum_{k=1}^{K}log(\Gamma(x_{k}))
        tmp = 0.0
        for i in x:
            left += loggamma(i)
            tmp += i
        right = loggamma(tmp) # log(\Gamma(\sum_{k=1}^{K}x_{k}))
        return left - right # log(\Delta(\vec{x}))

    def find_best_alpha(self, n, alphas):
        '''
        :param n: [float list] counts of words in this corpus
        :param alphas: [float list] e.g. [0.01, 0.02, 0.03]
        :return: the alpha which makes corpus lowest perplexity among all alphas
        '''
        scores = [self.cal_corpus_perplexity(self.cal_corpus_log_likelihood_12_30(n, [alpha for i in n]), self.train) for alpha in alphas]
        return alphas[scores.index(min(scores))]


if __name__ == '__main__':
    data = PTB(path="/home/sean/Desktop/chapter12/data/ptb")
    lm = BayesianUnigramLM(data)
    n_train = data.cal_counts_of_words(data.train)
    n_test = data.cal_counts_of_words(data.test)
    # print(n_valid)
    best_alpha = lm.find_best_alpha(n_train, [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7])
    print("the alpha which makes train corpus lowest perplexity is {}".format(best_alpha))
    best_alpha = [best_alpha for i in n_train]
    E_theta = lm.cal_E_theta(n_train, best_alpha)
    log_likelihood_test = lm.cal_corpus_log_likelihood_2_2(n_test, E_theta)
    print("the log_likelihood of test set is {}".format(log_likelihood_test))
    print("the perplexity of test set is {}".format(lm.cal_corpus_perplexity(log_likelihood_test, lm.test)))

