from nltk import ngrams
from collections import defaultdict
from data import SST

"""
Author: Wenyu Du
Organization: Westlake University
Email: wenyudu@yahoo.com

This is a simple implementation of count-based ngram language model.

Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

class ngramLM():
    def __init__(self, data, ngram_size):

        # Step 1 : Create a NESTED DICTIONARY to store ngram CO-OCCURRENCE
        self.ngram_size = ngram_size

        # Take bigram LM as an example, assume the total occurance of word "said" in the corpus is 3 with 2 times
        # having the next word "hello" and 1 time with word "goodbye", so we have {...,said:{hello:2,goodbye:1},...} in self.ngramLM
        self.ngramLM = defaultdict(lambda: defaultdict(lambda: 0))

        # Step 2: Load data and count ngram CO-OCCURRENCE
        # load data X and label y
        X = data.train["sents"]

        for sentence in X: # iterate sentences over the corpus
            for ngram in ngrams(sentence, ngram_size): # enumerate ngrams and count co-occurrence

                # Again, exemplify this step with biagram LM, for sentence "said hello", self.ngramLM['said']['hello']+=1,
                # so {...,said:{hello:2,goodbye:1},...} => {...,said:{hello:3,goodbye:1},...}
                self.ngramLM[ngram[:-1]][ngram[-1]] += 1

    # Calculate probabilities
    def get_prob(self,sent):
        prob = 1
        for window in ngrams(sent.split(" "), self.ngram_size):
            prob *= (self.ngramLM[window[:-1]][window[-1]]) / (len(self.ngramLM[window[:-1]])) # calculate P(w)
        return prob

    # Calculate probabilities with add-one smoothing technique
    def get_prob_smoothing(self,sent):
        prob = 1
        for window in ngrams(sent.split(" "), self.ngram_size):
            total_count = float(sum(self.ngramLM[window[:-1]].values()))
            prob *= (self.ngramLM[window[:-1]][window[-1]] + 1) / (total_count + len(self.ngramLM[window[:-1]])) # calculate P(w) with add-one smoothing
        return prob

if __name__ == '__main__':
    data = SST(fine_grained=False)
    lm = ngramLM(data,ngram_size=2)
    sent1 = 'OOV this movie is good'
    sent2 = 'OOV the hidden Markov model'
    print('sent1: ',sent1)
    print('sent2: ',sent2,"\n","*"*50)
    # compare two sentences
    print('Without smoothing: sent1: {:.8f}, sent2: {:.8f}'.format(lm.get_prob(sent1),lm.get_prob(sent2)))
    # compare two sentences using algorithm with smoothing
    print('With smoothing: sent1: {:.8f}, sent2: {:.8f}'.format(lm.get_prob_smoothing(sent1),lm.get_prob_smoothing(sent2)))
