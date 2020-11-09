import os
import itertools
from collections import Counter

"""
Author: Zebin Ou
Organization: Westlake University
Email: simtony2@gmail.com
"""


class CoNLL2000_Chunking(object):
    def __init__(self):
        # load raw data
        if os.path.exists("data/train.txt"):
            self.train = self.load_data("data/train.txt")
        else:
            raise ValueError(
                    "'train.txt' not found, please download it from https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz and extract them in the same path of the code.")

        if os.path.exists("data/test.txt"):
            self.test = self.load_data("data/test.txt")
        else:
            raise ValueError(
                    "'test.txt' not found, please download it from http://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz and extract them in the same path of the code.")

        # Construct a map from word/chunk_tag to integers
        if not (os.path.exists("data/words.txt") and os.path.exists("data/chunk_tags.txt")):
            word2count = Counter()
            chunk_tag2count = Counter()
            for words, _, chunk_tags in itertools.chain(self.train, self.test):
                word2count.update(words)
                chunk_tag2count.update(chunk_tags)
            with open("data/words.txt", "w") as fout:
                for word, count in word2count.most_common():
                    fout.write("{}\t{}\n".format(word, count))
            with open("data/chunk_tags.txt", "w") as fout:
                for chunk_tag, count in chunk_tag2count.most_common():
                    fout.write("{}\t{}\n".format(chunk_tag, count))
            print("Vocabularies at 'data/words.txt' and 'data/chunk_tags.txt'.")

        self.words = self.load_token_list("data/words.txt")
        self.chunk_tags = self.load_token_list("data/chunk_tags.txt")

    def load_data(self, filename):
        data = []
        with open(filename, "r") as fin:
            words = []
            pos_tags = []
            chunk_tags = []
            for line in fin:
                line = line.strip()
                if line:
                    word, pos_tag, chunk_tag = line.strip().split()
                    words.append(word)
                    pos_tags.append(pos_tag)
                    chunk_tags.append(chunk_tag)
                else:
                    data.append((tuple(words), tuple(pos_tags), tuple(chunk_tags)))
                    words = []
                    pos_tags = []
                    chunk_tags = []
        return data

    def load_token_list(self, filename):
        with open(filename, "r") as fin:
            token_list = [line.strip().split()[0] for line in fin if line.strip()]
        return token_list


if __name__ == '__main__':
    conll = CoNLL2000_Chunking()
    print(conll.train[0])
    print(conll.test[0])
    print(conll.words)
    print(conll.chunk_tags)
